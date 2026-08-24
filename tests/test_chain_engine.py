"""Composition is only safe if the contract is enforced between stages."""

import numpy as np
import pytest

from earshot import engines
from earshot.testing import assert_engine_contract

RATE = 48000


def test_a_chain_applies_its_stages_in_order():
    chain = engines.load("chain:passthrough:6+passthrough:6").engine
    x = np.full(RATE, 0.01, dtype=np.float32)
    out = chain.process(x, RATE)
    # Two 6 dB stages are 12 dB, and in that order — not one, not three.
    assert out.mean() == pytest.approx(0.01 * 10 ** (12 / 20), rel=1e-4)


def test_the_name_says_what_it_is_made_of():
    chain = engines.load("chain:passthrough+passthrough:3").engine
    assert "→" in chain.name and "passthrough" in chain.name


def test_a_chain_keeps_the_contract():
    assert_engine_contract(engines.load("chain:passthrough+passthrough:3").engine)


def test_one_engine_is_not_a_chain():
    with pytest.raises(engines.EngineError) as caught:
        engines.load("chain:passthrough")
    assert "two or more" in str(caught.value)


def test_an_unknown_stage_is_reported_by_name():
    with pytest.raises(engines.EngineError) as caught:
        engines.load("chain:passthrough+nosuchengine")
    assert "nosuchengine" in str(caught.value)


def test_a_broken_stage_is_named_not_the_chain():
    """Otherwise debugging a chain means bisecting it by hand."""

    class Truncates:
        name = "truncates"

        def process(self, audio, rate):
            return np.asarray(audio)[:-1]

    good = engines.load("passthrough")
    chain = engines.engines_chain = engines.load("chain:passthrough+passthrough").engine
    chain.stages = [good, engines.Loaded(Truncates())]
    with pytest.raises(engines.EngineError) as caught:
        chain.process(np.zeros(1000, dtype=np.float32), RATE)
    assert "truncates" in str(caught.value)
    assert "changed the sample count" in str(caught.value)
