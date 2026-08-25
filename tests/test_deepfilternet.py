"""DeepFilterNet's wiring, without PyTorch.

CI has no extras, so what is testable here is the spec parsing and the shim.
The model itself is exercised by
`test_the_real_model_keeps_the_contract` below, which runs where the extra is
present and skips with a reason where it is not.
"""

import sys

import pytest

from earshot import engines
from earshot.engines import deepfilternet as dfn


def test_the_attenuation_limit_is_parsed_and_named():
    """A denoiser allowed unlimited attenuation silences the room, which is
    the failure this project cares about most — so the limit is reachable."""
    try:
        engine = engines.load("deepfilternet:12").engine
    except engines.EngineError as exc:
        pytest.skip(f"extra not installed: {exc}")
    assert engine.attenuation_db == 12.0
    assert "12" in engine.name


def test_a_bad_limit_is_refused_before_anything_loads():
    with pytest.raises(engines.EngineError) as caught:
        engines.load("deepfilternet:quietly")
    assert "attenuation limit" in str(caught.value)


def test_the_shim_puts_back_what_torchaudio_removed():
    """DeepFilterNet 0.5 imports torchaudio.backend.common, which no longer
    exists. Pinning an old torchaudio would drag an old torch with it."""
    for name in ("torchaudio.backend.common", "torchaudio.backend"):
        sys.modules.pop(name, None)
    dfn._shim_torchaudio()
    from torchaudio.backend.common import AudioMetaData  # noqa: PLC0415

    meta = AudioMetaData(sample_rate=48000)
    assert meta.sample_rate == 48000


def test_the_shim_is_idempotent():
    dfn._shim_torchaudio()
    first = sys.modules["torchaudio.backend.common"]
    dfn._shim_torchaudio()
    assert sys.modules["torchaudio.backend.common"] is first


def test_the_model_rate_is_the_rate_everything_else_uses():
    """No resampling, so the length is exact without correction."""
    assert dfn.MODEL_RATE == 48000


def test_the_real_model_keeps_the_contract():
    """As in test_lavasr: the assertion this file's header already claimed.

    It has never run anywhere, because the extra is installed on no machine
    that has run these tests. It skips with a reason until one does, which is
    the repo's rule — never silently.
    """
    try:
        loaded = engines.load("deepfilternet")
    except engines.EngineError as exc:
        pytest.skip(f"deepfilternet unavailable here: {exc}")

    from earshot.testing import assert_engine_contract

    assert_engine_contract(loaded.engine)
