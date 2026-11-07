"""NovaSR's wiring, without PyTorch or the weights.

The model is 53 KB, but CI still has no torch, so what is testable here is
the pinning and the spec. The model itself is exercised by
`test_the_real_model_keeps_the_contract`, which skips with a reason where the
extra or the weights are missing.
"""

import pytest

from earshot import engines, fetch
from earshot.engines import novasr


def test_the_weights_are_pinned_to_a_commit():
    (asset,) = novasr.ASSETS
    assert len(asset.sha256) == 64
    assert "/resolve/main/" not in asset.url
    # v1 is what upstream's FastSR() loads when given no path.
    assert asset.url.endswith("/pytorch_model_v1.bin")


def test_it_takes_no_options():
    with pytest.raises(engines.EngineError) as caught:
        engines.load("novasr:fast")
    assert "no options" in str(caught.value)


def test_no_download_names_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path))
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    with pytest.raises(engines.EngineError) as caught:
        fetch.ensure_all(novasr.ASSETS)
    assert "pytorch_model_v1.bin" in str(caught.value)


def test_the_real_model_keeps_the_contract():
    try:
        loaded = engines.load("novasr")
    except engines.EngineError as exc:
        pytest.skip(f"novasr unavailable here: {exc}")

    from earshot.testing import assert_engine_contract

    assert_engine_contract(loaded.engine)
