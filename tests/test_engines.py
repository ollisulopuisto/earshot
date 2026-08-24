"""The contract is the only thing the bench cannot verify after the fact."""

import numpy as np
import pytest

from earshot import engines


def test_registry_knows_its_schemes():
    assert "passthrough" in engines.schemes()
    assert "vst3" in engines.schemes()


def test_unknown_scheme_lists_the_known_ones():
    with pytest.raises(engines.EngineError) as caught:
        engines.load("magic:/x")
    assert "passthrough" in str(caught.value)


def test_a_path_with_colons_still_loads():
    """The scheme is the first colon, not every colon."""
    with pytest.raises(engines.EngineError) as caught:
        engines.load("vst3:/Volumes/Odd:Name/Thing.vst3")
    assert "no plug-in at" in str(caught.value)


@pytest.mark.parametrize(
    "bad, message",
    [
        (lambda x: x[:-1], "changed the sample count"),
        (lambda x: np.stack([x, x]), "channels"),
        (lambda x: x * np.inf, "non-finite"),
    ],
)
def test_contract_violations_are_named(bad, message):
    x = np.zeros(1000, dtype=np.float32) + 0.1
    with pytest.raises(engines.EngineError) as caught:
        engines.check_contract(x, bad(x), "thing")
    assert message in str(caught.value)


def test_passthrough_gain_is_applied():
    x = np.full(1000, 0.1, dtype=np.float32)
    out = engines.load("passthrough:6").engine.process(x, 48000)
    assert out.mean() == pytest.approx(0.1 * 10 ** (6 / 20), rel=1e-4)
