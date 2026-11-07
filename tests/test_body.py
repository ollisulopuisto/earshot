"""Putting body back: the lowest harmonics a telephone band removed.

Pulakka et al. (Interspeech 2011; IEEE TASLP 2012) extend telephone speech
below 300 Hz by synthesising the missing harmonics from the pitch. These
tests hold the parts that can be wrong without anyone hearing it at once:
the pitch, the amount, and the promise to leave everything else alone.
"""

import numpy as np
import pytest
from scipy import signal

from earshot import engines, metrics
from earshot.engines import body

RATE = 48000


def _voiced(seconds=3.0, f0=120.0):
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 30))
    return (0.1 * x / np.abs(x).max()).astype(np.float32)


def _phone(x):
    sos = signal.butter(8, 300.0, btype="high", fs=RATE, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


@pytest.mark.parametrize("f0", [95.0, 120.0, 210.0])
def test_the_pitch_is_found_without_the_fundamental(f0):
    """The fundamental is exactly what the telephone removed; the period is
    still there in the harmonics above it."""
    track = body.pitch_track(_phone(_voiced(f0=f0)), RATE)
    voiced = track[track > 0]
    assert len(voiced) > 0.8 * len(track)
    assert np.median(voiced) == pytest.approx(f0, rel=0.03)


def test_the_missing_band_comes_back_to_the_balance_it_is_aimed_at(monkeypatch):
    """With the target set to the original's own balance, the repair should
    land near it. Which balance real speech has is a measurement, kept in
    TARGET_DB; this checks the machinery, not the calibration."""
    original = _voiced()
    monkeypatch.setattr(body, "target_db", lambda f0: metrics.body(original, RATE))
    phone = _phone(original)
    restored = body.BodyEngine().process(phone, RATE)
    assert metrics.body(phone, RATE) < metrics.body(original, RATE) - 15
    assert metrics.body(restored, RATE) == pytest.approx(metrics.body(original, RATE), abs=3.0)


def test_only_the_low_band_is_touched():
    phone = _phone(_voiced())
    restored = body.BodyEngine().process(phone, RATE)
    sos = signal.butter(8, 400.0, btype="high", fs=RATE, output="sos")
    above = signal.sosfiltfilt(sos, restored - phone)
    assert np.sqrt(np.mean(above**2)) < 1e-3 * np.sqrt(np.mean(phone**2))


def test_a_voice_that_has_its_body_is_left_alone():
    x = _voiced()
    np.testing.assert_array_equal(body.BodyEngine().process(x, RATE), x)


def test_it_keeps_the_contract():
    from earshot.testing import assert_engine_contract

    assert_engine_contract(engines.load("body").engine)


def test_the_cutoff_is_a_frequency():
    assert engines.load("body:250").engine.cutoff == 250.0
    with pytest.raises(engines.EngineError):
        engines.load("body:low")
