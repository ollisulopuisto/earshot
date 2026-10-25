"""The engine for the buzz, and the control every denoiser should beat.

Case two of this project is a good local microphone with something wrong that
is not the voice: a live room, or a constant buzz from a ground loop. LavaSR
cannot touch either — it crossfades its output in above 4 kHz, and a mains
family lives between 50 and 300 Hz. So the bench had no engine at all for the
use case, and a classical notch is both the right tool and the honest
baseline: if a neural denoiser cannot beat it on a ground loop, the ground
loop did not need a neural denoiser.
"""

import numpy as np

from earshot import degrade, engines, metrics, probes
from earshot.testing import assert_engine_contract

RATE = 48000


def test_it_keeps_the_contract():
    assert_engine_contract(engines.load("notch:50").engine)


def test_it_removes_the_buzz_and_leaves_the_voice():
    """Measured where the hum is actually visible, which is not everywhere.

    The first version of this test summed a band around all six harmonics and
    reported 0.5 dB of removal from an engine that was taking out 21 to 37 dB.
    At 200, 250 and 300 Hz the voice already carries more energy than the hum
    adds, so those bands measure speech and drown the result. The engine
    leaves those harmonics alone on purpose; the test now asks only about the
    ones it says it acted on.
    """
    clean = probes.default_material(RATE, 8.0)
    buzzing = degrade.by_name("ground-loop").apply(clean, RATE)

    engine = engines.load("notch:50").engine
    fixed = engine.process(buzzing, RATE)

    assert len(engine.notched) >= 3, f"only found {engine.notched}"

    def at(x, freq):
        f = np.fft.rfftfreq(len(x), 1 / RATE)
        p = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
        m = (f > freq - 1) & (f < freq + 1)
        return 10 * np.log10(p[m].sum() + 1e-30)

    for freq in engine.notched:
        removed = at(buzzing, freq) - at(fixed, freq)
        assert removed > 15, f"{freq:.0f} Hz: only took out {removed:.1f} dB"

    # The speech has to survive it. This is the failure mode of a notch set
    # too wide or too deep: it eats the fundamental of a low voice.
    kept = metrics.band_correlation(clean, fixed, RATE, 200.0, 4000.0)
    assert kept > 0.98, f"speech band correlation fell to {kept:.3f}"


def test_it_leaves_harmonics_the_voice_is_already_using():
    """A harmonic buried under speech is not worth the damage of removing it."""
    clean = probes.default_material(RATE, 8.0)
    buzzing = degrade.by_name("ground-loop").apply(clean, RATE)
    engine = engines.load("notch:50").engine
    engine.process(buzzing, RATE)
    assert 50.0 in engine.notched, "missed the fundamental"
    assert all(f <= 300.0 for f in engine.notched)


def test_a_recording_without_hum_is_barely_touched():
    """The router's lesson, applied here: doing nothing must cost nothing."""
    clean = probes.default_material(RATE, 8.0)
    engine = engines.load("notch:50").engine
    out = engine.process(clean, RATE)
    assert engine.notched == [], f"notched {engine.notched} on material with no hum"
    assert metrics.band_correlation(clean, out, RATE, 200.0, 4000.0) > 0.999


def test_the_mains_frequency_is_selectable():
    """50 Hz is Europe. A recording from the US needs 60, and asking for
    something absurd must fail loudly rather than quietly do nothing."""
    assert engines.load("notch:60").engine.name.endswith("60Hz)")
    try:
        engines.load("notch:nonsense")
    except engines.EngineError as exc:
        assert "number" in str(exc).lower()
    else:
        raise AssertionError("a non-numeric mains frequency was accepted")
