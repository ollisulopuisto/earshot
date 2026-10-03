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


@pytest.mark.parametrize("name", ["ears", "podcast", "bwe", "bwe-ears"])
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


def test_one_at_a_time_frees_each_engine_before_the_next(tmp_path, monkeypatch):
    """On a 15 GB T4 a set ran out of memory with UniverSR and UniPASE both
    held; AudioSR alone is a 6.2 GB checkpoint. One at a time loads each
    engine for its take and closes it before the next is loaded."""
    import numpy as np
    import soundfile as sf

    script = _script()
    loads, closes = [], []

    class Engine:
        name = "counted"

        def process(self, audio, rate):
            return audio

        def close(self):
            closes.append(1)

    def fake_load(spec):
        loads.append(spec)
        return engines.Loaded(Engine())

    monkeypatch.setattr(script.engines, "load", fake_load)
    monkeypatch.setattr(script.listen, "build", lambda *a, **k: None)
    sf.write(tmp_path / "voice.wav", np.zeros(48000 * 10, dtype=np.float32), 48000)
    takes = [("counted", "A", ""), ("counted:2", "B", "")]
    plan = script.Plan(
        [("01", "voice.wav", 0.0, "clean", "t", "l", takes),
         ("02", "voice.wav", 0.0, "clean", "t", "l", takes)],
        ("o", ""), "", "title",
    )

    script.main(tmp_path, tmp_path / "out", plan, one_at_a_time=True)
    assert len(loads) == 4 and len(closes) == 4

    loads.clear(), closes.clear()
    script.main(tmp_path, tmp_path / "out2", plan)
    assert len(loads) == 2  # cached across comparisons, as before
