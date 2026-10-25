"""Tone match: an engine's output pulled back to the speaker's own balance.

The owner liked Sidon but heard "speaker tone changes". Measured on twelve
pp53 comparisons (2026-10-04), Sidon against the original, relative to
300-3000 Hz: body +2.0..+2.5 dB, mid 800-2500 Hz +2.0..+3.4, presence and
above -1..-3.4. A long-term EQ match to the input — trustworthy below its
band edge — can undo a balance shift; it cannot undo the steadier pitch.
"""

import numpy as np
import pytest
from scipy import signal

from earshot import engines
from earshot.engines import tonematch

RATE = 48000


def _voice(seconds=4.0):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(int(seconds * RATE))
    x = signal.lfilter([1.0], [1.0, -0.9], x)
    return (0.1 * x / np.abs(x).max()).astype(np.float32)


def _band_db(x, low, high):
    f, p = signal.welch(x, RATE, nperseg=8192)
    return 10 * np.log10(p[(f >= low) & (f < high)].sum())


class Tilt:
    """Stands in for an engine that adds mid and darkens the top."""

    name = "tilt"

    def process(self, audio, rate):
        sos_mid = signal.butter(2, (800, 2500), btype="band", fs=rate, output="sos")
        sos_top = signal.butter(2, 3000, btype="high", fs=rate, output="sos")
        # About -3 dB above 3 kHz, inside the 1-3.4 dB Sidon measured; a 6 dB
        # cut sat exactly on the match's +-6 dB limit and could not come back.
        y = audio + 0.6 * signal.sosfiltfilt(sos_mid, audio) - 0.3 * signal.sosfiltfilt(sos_top, audio)
        return y.astype(np.float32)


def test_the_balance_comes_back():
    x = _voice()
    shifted = Tilt().process(x, RATE)
    matched = tonematch.ToneMatchEngine(Tilt()).process(x, RATE)

    def mid_vs_speech(a):
        return _band_db(a, 800, 2500) - _band_db(a, 300, 3000)

    assert abs(mid_vs_speech(shifted) - mid_vs_speech(x)) > 1.0
    assert abs(mid_vs_speech(matched) - mid_vs_speech(x)) < 0.5
    top = _band_db(matched, 4000, 8000) - _band_db(matched, 300, 3000)
    assert top == pytest.approx(_band_db(x, 4000, 8000) - _band_db(x, 300, 3000), abs=1.0)


def test_above_the_inputs_band_edge_it_trusts_the_engine():
    """A telephone input has nothing above 3.4 kHz to match against; the
    engine's invented top must be left as the engine made it."""
    x = _voice()
    phone = signal.sosfiltfilt(signal.butter(8, 3400, fs=RATE, output="sos"), x).astype(np.float32)

    class Brighten:
        name = "brighten"

        def process(self, audio, rate):
            return np.asarray(x, dtype=np.float32)  # "restores" the full band

    matched = tonematch.ToneMatchEngine(Brighten()).process(phone, RATE)
    assert _band_db(matched, 6000, 12000) == pytest.approx(_band_db(x, 6000, 12000), abs=1.0)


def test_it_loads_and_keeps_the_contract():
    loaded = engines.load("tonematch:passthrough")
    assert loaded.engine.name == "tonematch(passthrough)"
    from earshot.testing import assert_engine_contract

    assert_engine_contract(loaded.engine)
