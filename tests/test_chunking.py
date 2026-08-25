"""A whole episode has to fit in memory, and come out the same.

Measured on this machine, fresh process per length: LavaSR peaks at 0.30 GB
for 5 s of audio and 1.41 GB for 160 s, which extrapolates to somewhere
between 10 and 25 GB for a 56 minute episode. The machine has 32 GB with
20.7 GB wired by a VM and local models, so ``earshot restore`` on a real
recording is killed before it finishes — and processing real recordings is
what the project is for.

Chunking fixes that, but only if the output is the same. These tests are the
"same" half; the memory half is arithmetic.
"""

import numpy as np
import pytest

from earshot import engines

RATE = 48000


class Lowpass:
    """An engine with memory, so chunk boundaries can actually go wrong.

    A stateless engine would pass any chunking scheme, including a broken
    one. This one carries a tail across samples, which is what makes the
    overlap necessary in the first place.
    """

    name = "lowpass"

    def process(self, audio, rate):
        from scipy import signal
        sos = signal.butter(4, 3000.0, "low", fs=rate, output="sos")
        return signal.sosfilt(sos, np.asarray(audio, dtype=np.float64)).astype(np.float32)


@pytest.mark.parametrize("seconds", [0.5, 3.0, 7.5, 11.0])
def test_chunking_keeps_every_sample(seconds):
    """The rule the whole bench rests on, applied to the chunker itself."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(seconds * RATE)).astype(np.float32) * 0.1
    y = engines.process_in_chunks(Lowpass(), x, RATE, chunk_seconds=2.0,
                                  overlap_seconds=0.25)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))


def test_a_passthrough_survives_chunking_exactly():
    """If the engine changes nothing, chunking must change nothing either."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(9.0 * RATE)).astype(np.float32) * 0.1

    class Identity:
        name = "identity"

        def process(self, audio, rate):
            return np.asarray(audio, dtype=np.float32)

    y = engines.process_in_chunks(Identity(), x, RATE, chunk_seconds=2.0,
                                  overlap_seconds=0.25)
    assert np.allclose(y, x, atol=1e-6)


def test_chunked_matches_whole_file():
    """The measurement that makes chunking trustworthy — for a linear engine.

    This bounds the chunker itself, which is the only thing it can bound. A
    real generative engine need not agree with its own whole-file output at
    all, and LavaSR does not: measured on a 3 minute recording, chunked
    against whole-file came to -22.0 dB, hop-aligned chunks to -25.6, and
    two halves butted together with no overlap whatsoever to -26.1. The
    crossfade was never the difference.

    The cause is that LavaSR is not linear and not local. Halving its input
    changes its output by -18.4 dB from half the original output, and the
    same twenty seconds preceded by exactly one hundred hops of silence comes
    out -27.2 dB different. It is deterministic — the same buffer twice is
    bit-identical at -300 dB — so this is the engine reading its whole input,
    not noise.

    So "chunked equals whole-file" is not an available criterion for such an
    engine, and asserting it here against a real model would only encode a
    falsehood. What is testable is that the chunker adds nothing of its own,
    which is what a linear engine with a tail shows.
    """
    rng = np.random.default_rng(2)
    x = rng.standard_normal(int(12.0 * RATE)).astype(np.float32) * 0.1

    engine = Lowpass()
    whole = engine.process(x, RATE)
    chunked = engines.process_in_chunks(engine, x, RATE, chunk_seconds=2.0,
                                        overlap_seconds=0.25)

    error = 20 * np.log10(
        np.sqrt(np.mean((chunked - whole) ** 2)) / (np.sqrt(np.mean(whole ** 2)) + 1e-12)
        + 1e-12
    )
    assert error < -60, f"chunk seams left {error:.1f} dB of error"


def test_a_short_signal_is_not_chunked_at_all():
    """Nothing shorter than one chunk should take a different code path."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int(1.0 * RATE)).astype(np.float32) * 0.1
    engine = Lowpass()
    assert np.allclose(
        engines.process_in_chunks(engine, x, RATE, chunk_seconds=60.0,
                                  overlap_seconds=2.0),
        engine.process(x, RATE),
        atol=1e-6,
    )


def test_chunking_does_not_rewrite_the_recording_it_was_given():
    """The bug this nearly shipped with.

    An engine that returns its input unchanged returns a *view* of the
    caller's array, so fading the chunk in place rewrote the source. It left
    0.38 of error at every seam and would have quietly damaged the file being
    restored.
    """
    rng = np.random.default_rng(4)
    x = rng.standard_normal(int(9.0 * RATE)).astype(np.float32) * 0.1
    original = x.copy()

    class Identity:
        name = "identity"

        def process(self, audio, rate):
            return np.asarray(audio, dtype=np.float32)

    engines.process_in_chunks(Identity(), x, RATE, chunk_seconds=2.0,
                              overlap_seconds=0.25)
    assert np.array_equal(x, original), "the chunker modified its input"
