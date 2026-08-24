"""The bench has to be right about the easy cases before it is trusted on
the hard ones.

Passthrough is the control throughout: an engine that does nothing must
score as doing nothing. A bench that rewards it, or punishes it, is
measuring itself.
"""

import numpy as np
import pytest

from earshot import degrade, engines, probes

RATE = 48000


@pytest.fixture(scope="module")
def clean():
    return probes.default_material(RATE, 6.0)


def test_passthrough_recovers_nothing(clean):
    run = probes.run_all(engines.load("passthrough"), clean, RATE,
                         degrade.by_name("narrowband-voip"))
    assert not run.failed and not run.skipped
    # It cannot have got the missing band back, because it did not try.
    assert abs(run.value("recovery", "gained")) < 0.01
    # And it is still exactly the signal it was given, in every band.
    for name, _, _ in probes.BANDS:
        r = run.value("origin", name)
        if r == r:  # not NaN
            assert r > 0.99, name


def test_passthrough_is_deterministic_and_free(clean):
    run = probes.run_all(engines.load("passthrough"), clean, RATE,
                         degrade.by_name("clean"))
    assert run.value("stability", "overall") > 200
    assert run.value("stability", "low") > 100
    assert run.value("throughput", "realtime") > 10


def test_a_gain_stage_is_not_a_restoration(clean):
    """Turning the signal up must not look like cleaning it.

    This is the trap the cleanup probe exists to avoid: a naive
    signal-to-noise reading rewards any gain change.
    """
    run = probes.run_all(engines.load("passthrough:6"), clean, RATE,
                         degrade.by_name("hiss"))
    assert run.value("cleanup", "floor_change") == pytest.approx(6.0, abs=0.5)
    assert run.value("cleanup", "speech_change") == pytest.approx(6.0, abs=0.5)
    # Both ends moved together, so no range was gained. That is the tell.
    assert abs(run.value("cleanup", "range_gained")) < 0.5


def test_recovery_band_follows_the_damage():
    """Scoring the whole spectrum would drown the part that was removed."""
    low, high = probes._damaged_band(degrade.by_name("narrowband-voip"), RATE)
    assert low == 3400.0 and high > low
    # Damage that is not band-limiting is scored across the board.
    low, high = probes._damaged_band(degrade.by_name("hiss"), RATE)
    assert low == 100.0


def test_a_broken_engine_is_reported_not_swallowed(clean):
    class Truncates:
        name = "truncates"

        def process(self, audio, rate):
            return np.asarray(audio)[: len(audio) // 2]

    run = probes.run_all(engines.Loaded(Truncates()), clean, RATE,
                         degrade.by_name("clean"))
    assert "changed the sample count" in run.failed
    assert not run.results


def test_missing_tool_is_a_skip_with_a_reason(clean, monkeypatch):
    monkeypatch.setattr(probes, "_have", lambda tool: False)
    run = probes.run_all(engines.load("passthrough"), clean, RATE,
                         degrade.by_name("opus-16k"))
    assert run.skipped == "needs ffmpeg"


def test_preservation_measures_non_speech():
    run = probes.preservation(engines.load("passthrough"), RATE, 3.0)
    assert abs(run.value("preservation", "overall")) < 1e-6
    assert run.value("preservation", "500-2000Hz") is not None
