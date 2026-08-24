"""The contract kit has to fail the engines it is meant to fail.

A kit that passes everything protects nothing, so each rule is checked
against an engine that breaks exactly that rule and nothing else.
"""

import numpy as np
import pytest

from earshot import engines
from earshot.testing import assert_engine_contract


def test_the_passthrough_control_passes():
    assert_engine_contract(engines.load("passthrough").engine)
    assert_engine_contract(engines.load("passthrough:3").engine)


class _Framed:
    """Rounds down to whole 1024-sample frames — the classic length bug."""

    name = "framed"

    def process(self, audio, rate):
        n = (len(audio) // 1024) * 1024
        return np.asarray(audio)[:n]


class _Hallucinates:
    """Invents speech-level noise where the input was digital silence."""

    name = "hallucinates"

    def process(self, audio, rate):
        rng = np.random.default_rng(0)
        return (np.asarray(audio) + 0.1 * rng.normal(0, 1, len(audio))).astype("float32")


class _Stateful:
    """Leaks the previous file's tail into the start of this one.

    The length is right and the samples are finite, so nothing in the
    contract catches it. This is what `reset=False` does to a real plug-in,
    and it is silent.
    """

    name = "stateful"

    def __init__(self):
        self.tail = None

    def process(self, audio, rate):
        out = np.asarray(audio, dtype=np.float32).copy()
        if self.tail is not None:
            n = min(len(self.tail), len(out))
            out[:n] += self.tail[:n]
        self.tail = out[-2048:].copy()
        return out


class _Crashes:
    name = "crashes"

    def process(self, audio, rate):
        if len(audio) < 1000:
            raise ValueError("too short for my window")
        return np.asarray(audio)


@pytest.mark.parametrize(
    "engine, message",
    [
        (_Framed(), "changed the sample count"),
        (_Hallucinates(), "invention, not restoration"),
        (_Stateful(), "carrying state between calls"),
        (_Crashes(), "raised on"),
    ],
    ids=lambda e: getattr(e, "name", ""),
)
def test_each_rule_catches_its_own_breakage(engine, message):
    with pytest.raises(AssertionError) as caught:
        assert_engine_contract(engine)
    assert message in str(caught.value)
