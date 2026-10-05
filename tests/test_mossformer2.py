"""MossFormer2 SE 48K: the discriminative front end the research survey asked for.

ClearerVoice-Studio's 48 kHz enhancer (Apache-2.0 code and weights). It
predicts a mask over the input's own STFT, so — unlike the resynthesisers —
the speech it keeps is the speaker's own waveform. These tests hold that
property, the contract, and determinism: upstream's filterbank adds random
dither on every call.
"""

import numpy as np
import pytest
from scipy import signal

from earshot import degrade, engines, fetch
from earshot.engines import mossformer2

RATE = 48000


def test_the_weights_are_pinned():
    (asset,) = mossformer2.ASSETS
    assert len(asset.sha256) == 64 and "/resolve/main/" not in asset.url


def test_no_download_names_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path))
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    with pytest.raises(engines.EngineError) as caught:
        fetch.ensure_all(mossformer2.ASSETS)
    assert "last_best_checkpoint.pt" in str(caught.value)


def _engine():
    try:
        return engines.load("mossformer2").engine
    except engines.EngineError as exc:
        pytest.skip(f"mossformer2 unavailable: {exc}")


def test_the_real_model_keeps_the_contract():
    from earshot.testing import assert_engine_contract

    assert_engine_contract(_engine())


def test_it_is_the_same_twice():
    from earshot import probes

    engine = _engine()
    x = degrade.noise(probes.default_material(RATE, 3.0), RATE, snr_db=10.0)
    np.testing.assert_array_equal(engine.process(x, RATE), engine.process(x, RATE))


def test_the_speech_it_keeps_is_the_input():
    """A mask keeps the input's phase: in the speech band the output must
    still correlate with the clean signal that went in."""
    from earshot import probes

    clean = probes.default_material(RATE, 3.0)
    y = _engine().process(degrade.noise(clean, RATE, snr_db=20.0), RATE)
    sos = signal.butter(6, (300.0, 3000.0), btype="band", fs=RATE, output="sos")
    a, b = signal.sosfiltfilt(sos, clean), signal.sosfiltfilt(sos, y)
    assert np.corrcoef(a, b)[0, 1] > 0.9
