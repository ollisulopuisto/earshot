"""UniverSR's wiring, without PyTorch or the weights.

CI has neither, so what is testable here is the spec, the pinning, and the
choice of band — which is where this engine can quietly do the wrong thing:
UniverSR low-passes its input to the band it is told, so a band set too low
throws away speech that was there. The model itself is exercised by
`test_the_real_model_keeps_the_contract`, which skips with a reason where the
extra or the weights are missing.
"""

import numpy as np
import pytest
from scipy import signal

from earshot import engines, fetch
from earshot.engines import universr

RATE = 48000


def _speechlike(seconds: float = 3.0, ceiling_hz: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(seconds * RATE)
    x = rng.standard_normal(n)
    # Pink-ish tilt and a syllable envelope, so the speech band leads.
    x = signal.lfilter([1.0], [1.0, -0.95], x)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * np.arange(n) / RATE)
    x = x * envelope
    if ceiling_hz:
        sos = signal.butter(16, ceiling_hz, fs=RATE, output="sos")
        x = signal.sosfiltfilt(sos, x)
    return (0.1 * x / np.max(np.abs(x))).astype(np.float32)


def test_the_weights_are_pinned():
    assert universr.ASSETS
    for asset in universr.ASSETS:
        assert len(asset.sha256) == 64, asset.name
        # A branch name would let the file change under the digest's feet;
        # the URL names a commit so a mismatch means corruption, not news.
        assert "/resolve/main/" not in asset.url, asset.name
        assert asset.url.startswith("https://huggingface.co/"), asset.name
    # from_pretrained reads config.yaml and pytorch_model.bin from one folder.
    folders = {asset.name.rsplit("/", 1)[0] for asset in universr.ASSETS}
    assert len(folders) == 1


@pytest.mark.parametrize("argument", ["9", "48", "wide", "16@"])
def test_a_bad_band_is_refused_before_anything_loads(argument):
    with pytest.raises(engines.EngineError) as caught:
        engines.load(f"universr:{argument}")
    assert "8, 12, 16 or 24" in str(caught.value)


@pytest.mark.parametrize(
    "edge, rate",
    [(3400.0, 8000), (5000.0, 8000), (6100.0, 12000), (7500.0, 12000),
     (8000.0, 16000), (11000.0, 16000), (14500.0, 24000)],
)
def test_the_band_never_sits_above_the_content(edge, rate):
    """UniverSR treats everything below the band as real and keeps it. A band
    above the content would hand it an empty strip to keep as silence; one
    below throws speech away — the lesser harm, and the router puts it back."""
    assert universr.input_rate_for(edge) == rate


def test_the_edge_of_a_telephone_band_is_found():
    edge = universr.content_edge(_speechlike(ceiling_hz=3400.0), RATE)
    assert 3000.0 < edge < 4500.0


def test_full_band_input_is_left_alone():
    """There is no missing band to extend, and the model would low-pass the
    top 12 kHz of a good microphone away to make room for its own."""
    x = _speechlike()
    assert universr.content_edge(x, RATE) >= universr.FULL_BAND_HZ
    engine = universr.UniverSREngine.__new__(universr.UniverSREngine)
    engine.fixed_rate = None
    engine.name = "universr"
    engine.model = None  # touching the model here would be the bug
    np.testing.assert_array_equal(engine.process(x, RATE), x)


def test_silence_is_left_alone():
    x = np.zeros(RATE, dtype=np.float32)
    assert universr.content_edge(x, RATE) >= universr.FULL_BAND_HZ


def test_no_download_names_both_files(tmp_path, monkeypatch):
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path))
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    with pytest.raises(engines.EngineError) as caught:
        fetch.ensure_all(universr.ASSETS)
    assert str(caught.value).count("curl -L") == len(universr.ASSETS)


def test_the_real_model_keeps_the_contract():
    """One Euler step without guidance: the kit makes 23 calls, and at
    upstream's sixteen model passes per call this test was on course for half
    an hour on an M1 Max CPU; like this it took 2 min 45 s. Length and alignment are decided by the wiring
    around the solver, which is the same either way."""
    try:
        engine = universr.UniverSREngine(
            16000, ode_steps=1, guidance=None, ode_method="euler"
        )
    except engines.EngineError as exc:
        pytest.skip(f"universr unavailable here: {exc}")

    from earshot.testing import assert_engine_contract

    assert_engine_contract(engine)


def test_gpu_memory_is_released_after_each_call(monkeypatch):
    """On a Colab T4 the second comparison ran out of memory with 3.44 GiB
    reserved by PyTorch but unallocated: every engine of a listening set
    stays loaded, and UniverSR's activations on 8 s are gigabytes."""
    torch = pytest.importorskip("torch")
    released = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(1))

    class Model:
        _device = "cuda"

        def enhance(self, x, **_):
            return x.clone()

    engine = universr.UniverSREngine.__new__(universr.UniverSREngine)
    engine.model, engine.fixed_rate, engine.name = Model(), 16000, "universr@16k"
    engine.ode_steps, engine.guidance, engine.ode_method = 1, None, "euler"
    engine.process(np.zeros(RATE, dtype=np.float32), RATE)
    assert released
