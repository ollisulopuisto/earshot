"""AudioSR's wiring, without AudioSR.

AudioSR needs numpy<=1.23.5, librosa 0.9.2 and transformers 4.30.2, none of
which install beside this project, so it runs in its own Python and the
engine talks to it over a pipe. What CI can check is that pipe: a stand-in
worker that speaks the same protocol, returns too many samples like the real
one pads to its 2.5 s grid, and has to come out of the engine keeping the
contract anyway.
"""

import sys
import textwrap

import pytest

from earshot import engines
from earshot.engines import audiosr

FAKE_WORKER = textwrap.dedent(
    """
    import sys
    import numpy as np
    import soundfile as sf

    print("ready", flush=True)
    for line in sys.stdin:
        source, target = line.rstrip("\\n").split("\\t")
        x, rate = sf.read(source, dtype="float32")
        # The real model works on a padded grid and returns 48 kHz; return
        # more than was asked for, as it does.
        y = np.concatenate([x, np.zeros(1234, dtype=np.float32)])
        sf.write(target, y, 48000, subtype="FLOAT")
        print("ok", flush=True)
    """
)


def test_the_weights_are_pinned_to_a_commit():
    (asset,) = audiosr.ASSETS
    assert len(asset.sha256) == 64
    assert "/resolve/main/" not in asset.url
    assert asset.url.endswith("/pytorch_model.bin")


def test_without_its_own_python_it_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv("EARSHOT_AUDIOSR_PYTHON", raising=False)
    with pytest.raises(engines.EngineError) as caught:
        engines.load("audiosr")
    assert "EARSHOT_AUDIOSR_PYTHON" in str(caught.value)


def test_a_bad_step_count_is_refused():
    with pytest.raises(engines.EngineError) as caught:
        engines.load("audiosr:many")
    assert "DDIM steps" in str(caught.value)


def test_the_pipe_keeps_the_contract(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(audiosr, "WORKER", worker)
    monkeypatch.setattr(audiosr, "ensure_all", lambda assets: [tmp_path / "unused.bin"])
    monkeypatch.setenv("EARSHOT_AUDIOSR_PYTHON", sys.executable)

    engine = engines.load("audiosr").engine
    try:
        from earshot.testing import assert_engine_contract

        assert_engine_contract(engine)
    finally:
        engine.close()
    assert engine.process_handle is None
