"""The listening set's plans name only recipes and engines that exist.

A typo in a plan is found by the script only after it has rendered every
comparison before it, which on the podcast set is minutes of DeepFilterNet.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from earshot import degrade, engines

SCRIPT = Path(__file__).parent.parent / "scripts" / "listening_set.py"


def _script():
    spec = importlib.util.spec_from_file_location("listening_set", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules; unregistered, it
    # finds None there and fails on Python 3.13.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _specs(spec):
    # A chain names its links after the colon, joined by "+".
    scheme, _, argument = spec.partition(":")
    if scheme == "chain":
        for link in argument.split("+"):
            yield from _specs(link)
    elif scheme in ("router", "keepzero"):
        yield scheme
        yield from _specs(argument)
    else:
        yield scheme


@pytest.mark.parametrize("name", ["ears", "podcast"])
def test_every_plan_resolves(name):
    plan = _script().SETS[name]
    assert plan.comparisons
    folders = [c[0] for c in plan.comparisons]
    assert len(folders) == len(set(folders))
    for _, _, start, recipe, _, _, takes in plan.comparisons:
        assert start >= 0
        if not isinstance(recipe, degrade.Damage):
            degrade.by_name(recipe)
        assert takes
        for spec, _, _ in takes:
            for scheme in _specs(spec):
                assert scheme in engines.schemes(), spec
