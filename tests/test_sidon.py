"""Sidon's wiring, without the gigabyte of weights.

Sidon (sarulab-speech, MIT) is the research survey's first pick: 48 kHz,
self-reported speaker similarity above Miipher's. Like UniPASE it re-speaks
the signal from SSL features, so `origin` reads near zero by design and the
waveform cannot be compared sample for sample. What can be checked is that
the speech stays where it was in time: the envelope of the output must line
up with the envelope of the input.
"""

import numpy as np
import pytest
from scipy import signal

from earshot import degrade, engines, fetch
from earshot.engines import sidon

RATE = 48000


def test_the_weights_are_pinned_to_a_commit():
    names = {asset.name.rsplit("/", 1)[1] for asset in sidon.ASSETS}
    assert {"feature_extractor_cpu.pt", "decoder_cpu.pt", "preprocessor_config.json"} <= names
    for asset in sidon.ASSETS:
        assert len(asset.sha256) == 64, asset.name
        assert "/resolve/main/" not in asset.url, asset.name


def test_it_takes_no_options():
    with pytest.raises(engines.EngineError):
        engines.load("sidon:fast")


def test_no_download_names_the_files(tmp_path, monkeypatch):
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path))
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    with pytest.raises(engines.EngineError) as caught:
        fetch.ensure_all(sidon.assets_for("cpu"))
    assert "feature_extractor_cpu.pt" in str(caught.value)


def _engine():
    try:
        return engines.load("sidon").engine
    except engines.EngineError as exc:
        pytest.skip(f"sidon unavailable here: {exc}")


def test_the_real_model_keeps_the_contract():
    from earshot.testing import assert_engine_contract

    assert_engine_contract(_engine())


def test_the_speech_stays_where_it_was():
    """A resynthesiser can return the right number of samples and still
    shift the speech; the envelope's cross-correlation peak must sit within
    10 ms of zero."""
    engine = _engine()
    from earshot import probes

    x = probes.default_material(RATE, 4.0)
    damaged = degrade.by_name("narrowband-voip").apply(x, RATE)
    y = engine.process(damaged, RATE)

    def envelope(a):
        sos = signal.butter(4, 30.0, fs=RATE, output="sos")
        return signal.sosfiltfilt(sos, np.abs(a))

    a, b = envelope(damaged), envelope(y)
    lags = signal.correlation_lags(len(b), len(a))
    lag = lags[np.argmax(signal.correlate(b - b.mean(), a - a.mean(), method="fft"))]
    assert abs(lag) < 0.010 * RATE, f"speech moved by {lag / RATE * 1000:.1f} ms"
