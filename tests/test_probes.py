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


def test_recovery_is_scored_where_a_hum_actually_lives():
    """The band the probe looks in has to be the band the damage hit.

    ``ground-loop`` puts its energy at 50 Hz and a few harmonics, and the
    default recovery band starts at 100 Hz — so the fundamental, the loudest
    part of it, was outside the measurement entirely, and the harmonics that
    were inside got diluted across 100 Hz to 16 kHz. An engine that removed
    21 to 37 dB of hum scored +0.00 dB of recovery.
    """
    from earshot import degrade, engines, probes

    rate = 48000
    low, high = probes._damaged_band(degrade.by_name("ground-loop"), rate)
    assert low < 50.0, f"band starts at {low:.0f} Hz, above the mains fundamental"
    assert high < 2000.0, f"band reaches {high:.0f} Hz and dilutes the hum away"

    clean = probes.default_material(rate, 8.0)
    run = probes.run_all(engines.load("notch:50"), clean, rate,
                         degrade.by_name("ground-loop"))
    gained = run.value("recovery", "gained")
    assert gained is not None and gained > 1.0, (
        f"a notch that removes the hum scored {gained} dB of recovery"
    )


def _voiced(seconds=6.0, f0=120.0):
    """A harmonic series with real energy at the fundamental, unlike
    `default_material`, which sits 15.6 dB down in 80-250 Hz already."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 30))
    return (0.1 * x / np.abs(x).max()).astype(np.float32)


def test_body_sees_a_telephone_band_and_passthrough_does_not_fix_it():
    """The owner heard restorations without body; measured on the listening
    sets, phone-band damage took 80-250 Hz about 20 dB down against the
    speech band and no engine put it back. The probe has to see that, and
    must not credit an engine that only passed it through."""
    run = probes.run_all(engines.load("passthrough"), _voiced(), RATE,
                         degrade.by_name("narrowband-voip"))
    assert run.value("body", "before") < -10
    assert run.value("body", "after") == pytest.approx(run.value("body", "before"), abs=0.01)
    assert abs(run.value("body", "restored")) < 0.01


def test_body_is_a_balance_not_a_level(clean):
    """Turning up the whole signal changes no balance."""
    run = probes.run_all(engines.load("passthrough:6"), clean, RATE,
                         degrade.by_name("clean"))
    assert abs(run.value("body", "after")) < 0.1


def test_removing_non_speech_counts_as_better():
    """The owner's goal, stated 2026-10-03: for podcast production only the
    voice is wanted — noise, hum, buzz and room sound are all to go. The
    probe used to score keeping non-speech as better; it is the reverse."""
    run = probes.preservation(engines.load("passthrough"))
    overall = [r for r in run.results if r.metric == "overall"][0]
    assert overall.better == "higher"
