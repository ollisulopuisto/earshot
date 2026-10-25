"""Damage that sounds like what happens to a podcast guest, not like a filter.

The owner's verdict on the synthetic calls (2026-10-03): not believable.
`narrowband-voip` band-limits and then cuts tapered holes in the signal; a
real VoIP call has none of those holes, because the receiving decoder
conceals a lost packet by extrapolating the last one. These recipes put a
real Opus encoder and decoder in the path and lose packets in bursts, and
model the two other complaints: a landline, and a guest who blew their
input gain past 0 dB.
"""

import numpy as np
import pytest
from scipy import signal

from earshot import degrade, probes

RATE = 48000

needs_opus = pytest.mark.skipif(
    not probes._have("ffmpeg+libopus"), reason="needs ffmpeg and libopus"
)


@pytest.fixture(scope="module")
def clean():
    return probes.default_material(RATE, 4.0)


def test_bursty_loss_hits_its_rate_and_comes_in_bursts():
    """Five per cent on average, in runs averaging 2.5 packets — a burst
    model, because a network that loses one packet usually loses the next."""
    lost = degrade.bursty_loss(200_000, loss=0.05, burst=2.5, seed=1)
    assert 0.045 < lost.mean() < 0.055
    edges = np.flatnonzero(np.diff(np.concatenate([[0], lost.astype(int), [0]])))
    runs = edges[1::2] - edges[::2]
    assert 2.2 < runs.mean() < 2.8


def test_bursty_loss_is_the_same_twice():
    a = degrade.bursty_loss(1000, loss=0.05, burst=2.5, seed=3)
    assert np.array_equal(a, degrade.bursty_loss(1000, loss=0.05, burst=2.5, seed=3))
    assert not np.array_equal(a, degrade.bursty_loss(1000, loss=0.05, burst=2.5, seed=4))


@needs_opus
def test_an_opus_call_keeps_length_and_alignment(clean):
    """No loss: the call must line up sample for sample with its source, or
    every number downstream measures the codec's delay."""
    y = degrade.opus_call(clean, RATE, loss=0.0)
    assert len(y) == len(clean)
    sos = signal.butter(8, 3000, fs=RATE, output="sos")
    a, b = signal.sosfiltfilt(sos, clean), signal.sosfiltfilt(sos, y)
    lags = signal.correlation_lags(len(a), len(b))
    lag = lags[np.argmax(signal.correlate(a, b, method="fft"))]
    # Exact on the whole band (measured 0); one sample below 3 kHz, because
    # Opus is not linear-phase and its low band runs 21 µs behind.
    assert abs(lag) <= 1
    assert degrade._lag(y, clean) == 0
    assert np.corrcoef(a, b)[0, 1] > 0.9


@needs_opus
def test_an_opus_call_is_wideband(clean):
    """The one real call measured here has a 7.5 kHz ceiling: Opus in its
    wideband mode, which stops at 8 kHz."""
    y = degrade.opus_call(clean, RATE, loss=0.0)
    f, p = signal.welch(y, RATE, nperseg=4096)
    speech = p[(f > 300) & (f < 3000)].mean()
    assert p[(f > 9000) & (f < 16000)].mean() < speech * 1e-4


@needs_opus
def test_lost_packets_are_concealed_not_zeroed(clean):
    """The decoder fills a lost packet from the one before; a hole of exact
    silence inside speech is the unbelievable version."""
    y = degrade.opus_call(clean, RATE, loss=0.05, seed=2)
    frame = RATE // 50
    frames = y[: len(y) // frame * frame].reshape(-1, frame)
    silent = np.all(frames == 0, axis=1).mean()
    assert silent < 0.01
    assert not np.allclose(y, degrade.opus_call(clean, RATE, loss=0.0))


@needs_opus
def test_an_opus_call_is_the_same_twice(clean):
    a = degrade.opus_call(clean, RATE, loss=0.05, seed=2)
    assert np.array_equal(a, degrade.opus_call(clean, RATE, loss=0.05, seed=2))


def test_a_landline_is_telephone_band_and_companded(clean):
    y = degrade.landline(clean, RATE)
    assert len(y) == len(clean)
    f, p = signal.welch(y, RATE, nperseg=4096)
    speech = p[(f > 500) & (f < 3000)].mean()
    assert p[(f > 4500) & (f < 16000)].mean() < speech * 1e-4
    assert p[(f > 30) & (f < 150)].mean() < speech * 1e-2


def test_overload_flattens_the_peaks_at_full_scale(clean):
    """Input gain pushed 12 dB past 0 dBFS: everything above the converter's
    ceiling is gone, and a run of samples sits exactly on it."""
    y = degrade.overload(clean, RATE, over_db=12.0)
    assert len(y) == len(clean)
    ceiling = np.abs(y).max()
    flat = np.mean(np.isclose(np.abs(y), ceiling, rtol=0, atol=1e-6))
    assert flat > 0.02
    # Turned back down afterwards, as anyone mixing the episode would.
    assert ceiling < np.abs(clean).max()


@pytest.mark.parametrize("name", ["voip-call", "overload", "overload-call", "landline"])
def test_the_new_recipes_exist_and_say_what_they_need(name):
    recipe = degrade.by_name(name)
    if "call" in name and name != "landline":
        assert recipe.needs == "ffmpeg+libopus"
