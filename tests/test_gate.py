"""A gate: restore only what is damaged, leave clean audio alone.

Both lead candidates harm clean podcast audio (pp53 bench, 2026-10-03:
UniPASE −3.55 dB, Sidon −5.41 dB of log-spectral distance on clean input).
The gate decides from cheap measurements whether the input needs a
generative engine at all. Its first detectors are the ones measured to
separate damage from clean pp53 audio — a band edge under 10 kHz, flat-topped
clipping, gated digital silence — and it is honest that hiss and room
reverberation are invisible to all three.
"""

import numpy as np
import pytest
from scipy import signal

from earshot import degrade, engines
from earshot.engines import gate

RATE = 48000
EARS_CLEAN = "out/colab-calls/out/kuuntelu-bwe-ears/01-puhdas/00-alkuperainen.wav"


@pytest.fixture(autouse=True)
def _no_dnsmos_on_synthetic(request, monkeypatch):
    """DNSMOS rightly reads the synthetic test voice as poor speech; the
    mechanics are tested without it, and DNSMOS on real speech below."""
    if "real_speech" not in request.keywords:
        from earshot import quality

        monkeypatch.setattr(quality, "available", lambda: False)


def _voice(seconds=4.0):
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * RATE)) / RATE
    # Harmonics falling about 12 dB an octave, as a voice's do. With a flat
    # 1/k series, the clipping inside narrowband-voip regenerated harmonics up
    # to 12.9 kHz; real pp53 speech under the same damage reads 4.3-6.4 kHz.
    x = sum(np.sin(2 * np.pi * k * 130 * t) / k**2 for k in range(1, 120) if k * 130 < 20000)
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = x + 0.002 * rng.standard_normal(len(t))  # a room tone, so nothing is empty
    return (0.1 * x / np.abs(x).max()).astype(np.float32)


class Marker:
    """Stands in for a generative engine; says that it ran."""

    name = "marker"

    def __init__(self):
        self.calls = 0

    def process(self, audio, rate):
        self.calls += 1
        return np.asarray(audio, dtype=np.float32) * 0.5


def test_clean_full_band_audio_is_passed_through_untouched():
    inner = Marker()
    x = _voice()
    out = gate.GateEngine(inner).process(x, RATE)
    np.testing.assert_array_equal(out, x)
    assert inner.calls == 0


@pytest.mark.parametrize("damage", ["telephone band", "landline", "wideband-voip"])
def test_a_missing_band_sends_it_to_the_engine(damage):
    """A plain telephone band, not narrowband-voip: that recipe clips after
    band-limiting, and on this strongly harmonic test voice the clipping
    refills the top to 13.4 kHz — the detector's known blind spot, recorded
    in the gate's docstring. Real pp53 speech under it read 4.3-6.4 kHz."""
    inner = Marker()
    x = _voice()
    y = (degrade.band_limit(x, RATE, 300.0, 3400.0) if damage == "telephone band"
         else degrade.by_name(damage).apply(x, RATE))
    gate.GateEngine(inner).process(y, RATE)
    assert inner.calls == 1


def test_clipping_sends_it_to_the_engine():
    inner = Marker()
    gate.GateEngine(inner).process(degrade.overload(_voice(), RATE, 12.0), RATE)
    assert inner.calls == 1


def test_gated_silence_sends_it_to_the_engine():
    inner = Marker()
    x = _voice()
    x[: RATE] = 0.0
    gate.GateEngine(inner).process(x, RATE)
    assert inner.calls == 1


def test_the_reasons_are_reported():
    y = degrade.band_limit(_voice(), RATE, 300.0, 3400.0)
    assert "band" in " ".join(gate.reasons(y, RATE))
    assert gate.reasons(_voice(), RATE) == []


def test_it_loads_around_any_engine_and_keeps_the_contract():
    loaded = engines.load("gate:passthrough")
    assert loaded.engine.name == "gate(passthrough)"
    from earshot.testing import assert_engine_contract

    assert_engine_contract(loaded.engine)


def test_dnsmos_weights_are_pinned():
    from earshot import quality

    (asset,) = quality.ASSETS
    assert len(asset.sha256) == 64
    assert "/raw/master/" not in asset.url and "/raw/main/" not in asset.url


@pytest.mark.real_speech
def test_hiss_and_room_reach_the_engine_when_quality_is_available():
    """The first detectors were blind to both (0/5 on pp53). DNSMOS measured
    room at SIG 1.22-2.25 against clean 2.62-3.51, and hiss at BAK
    2.64-3.64 against clean 3.55-3.95 (five pp53 excerpts, 2026-10-04)."""
    from pathlib import Path

    import soundfile as sf

    from earshot import quality

    if not quality.available():
        pytest.skip("DNSMOS needs onnxruntime and its weights")
    path = Path(__file__).parent.parent / EARS_CLEAN
    if not path.exists():
        pytest.skip("needs a real clean EARS excerpt in out/")
    x, rate = sf.read(path, dtype="float32")
    assert gate.reasons(x, rate) == []
    assert any("background" in r for r in gate.reasons(degrade.noise(x, rate, snr_db=15.0), rate))
    assert any("speech quality" in r for r in gate.reasons(degrade.reverb(x, rate, rt60=0.6), rate))


def test_the_quality_detector_is_optional():
    """CI has no onnxruntime; the gate must still run on its other detectors
    (the autouse fixture plays the part of the missing model)."""
    assert gate.reasons(_voice(), RATE) == []
