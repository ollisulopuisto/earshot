"""UniPASE's wiring, without PyTorch or 2.2 GB of weights.

What is testable in CI is the pinning, the spec, and the espnet stand-in:
the vendored post-net imports espnet for one base class and one lookup, and
installing espnet for that would pull in a speech toolkit's worth of
dependencies. The model itself is exercised by
`test_the_real_model_keeps_the_contract`, which skips with a reason where the
extra or the weights are missing.
"""

import sys

import pytest

from earshot import engines, fetch
from earshot.engines import unipase


def test_the_weights_are_pinned_to_a_commit():
    names = {asset.name.rsplit("/", 1)[1] for asset in unipase.ASSETS}
    # The four upstream's inference loads; Vocoder_WavLM-L24.pt is for
    # training validation only and is 455 MB nobody here needs.
    assert names == {"DeWavLM-Omni.pt", "Adapter.pt", "Vocoder_DWO-L1.pt", "PostNet.pt"}
    for asset in unipase.ASSETS:
        assert len(asset.sha256) == 64, asset.name
        assert "/resolve/main/" not in asset.url, asset.name


def test_packet_loss_concealment_can_be_switched_off():
    """On by default, as upstream has it. Real podcast material measured
    0.0 to 0.3 per cent packet loss, so whether it helps or only invents is
    worth hearing both ways."""
    with pytest.raises(engines.EngineError) as caught:
        engines.load("unipase:plc-of")
    assert "noplc" in str(caught.value)


def test_the_espnet_stand_in_answers_what_the_postnet_asks():
    for name in list(sys.modules):
        if name.startswith("espnet2"):
            sys.modules.pop(name)
    torch = pytest.importorskip("torch")
    unipase._shim_espnet()
    from espnet2.enh.separator.abs_separator import AbsSeparator  # noqa: PLC0415
    from espnet2.torch_utils.get_layer_from_string import get_layer  # noqa: PLC0415

    assert issubclass(AbsSeparator, torch.nn.Module)
    # TFGridNet's only call: get_layer(activation)() with "prelu" by default.
    assert isinstance(get_layer("prelu")(), torch.nn.PReLU)
    with pytest.raises(ValueError):
        get_layer("no-such-layer")


def test_no_download_names_every_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path))
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    with pytest.raises(engines.EngineError) as caught:
        fetch.ensure_all(unipase.ASSETS)
    assert str(caught.value).count("curl -L") == len(unipase.ASSETS)


def test_the_real_model_keeps_the_contract_except_in_silence():
    """Every rule but the last holds; the last is a finding, not a bug.

    Given two seconds of digital zero, UniPASE returns −32.8 dBFS (measured
    2026-10-02, M1 Max CPU): a resynthesiser speaking into nothing. The kit
    checks silence last, so failing there and only there means length,
    alignment, finiteness and state all passed. Wrapping the engine to hide
    it would make the bench measure the wrapper; ``keepzero:unipase`` is that
    wrapper, named as such. If this starts passing, the model changed.
    """
    try:
        loaded = engines.load("unipase")
    except engines.EngineError as exc:
        pytest.skip(f"unipase unavailable here: {exc}")

    from earshot.testing import assert_engine_contract

    with pytest.raises(AssertionError, match="from digital silence"):
        assert_engine_contract(loaded.engine)
