"""A metric that cannot be checked is an opinion with a decimal point."""

import numpy as np

from earshot import metrics, probes

RATE = 48000


def test_lsd_is_zero_for_identical_signals():
    x = probes.default_material(RATE, 3.0)
    assert metrics.log_spectral_distance(x, x, RATE) == 0.0


def test_lsd_grows_with_damage():
    from earshot import degrade
    x = probes.default_material(RATE, 3.0)
    mild = degrade.band_limit(x, RATE, low=0, high=12000)
    harsh = degrade.band_limit(x, RATE, low=0, high=3400)
    assert (metrics.log_spectral_distance(mild, x, RATE, 3400, 16000)
            < metrics.log_spectral_distance(harsh, x, RATE, 3400, 16000))


def test_band_correlation_separates_filtering_from_invention():
    """The metric the whole bench turns on: is the output still the input?"""
    rng = np.random.default_rng(4)
    x = probes.default_material(RATE, 3.0)
    quieter = x * 0.5                      # a filter: same waveform
    invented = rng.normal(0, 0.1, len(x))  # a synthesiser: new waveform
    assert metrics.band_correlation(x, quieter, RATE, 500, 4000) > 0.99
    assert abs(metrics.band_correlation(x, invented, RATE, 500, 4000)) < 0.2


def test_dynamics_tells_cleaning_from_turning_down():
    x = probes.default_material(RATE, 6.0)
    turned_down = x * 0.1
    before, after = metrics.dynamics(x, RATE), metrics.dynamics(turned_down, RATE)
    # Turning down moves both ends together and opens no range at all.
    assert abs(after.range_db - before.range_db) < 1.0
    assert after.speech < before.speech - 15


def test_repeatability_is_a_margin_not_a_verdict():
    x = probes.default_material(RATE, 2.0)
    assert metrics.repeatability(x, x.copy()) > 200           # bit-identical
    assert 35 < metrics.repeatability(x, x * 1.01) < 45       # audible drift
    noise = np.random.default_rng(5).normal(0, 1e-4, len(x)).astype("float32")
    assert 50 < metrics.repeatability(x, x + noise) < 70      # inaudible drift


def test_lsd_ignores_a_pure_gain_change():
    """Otherwise every engine that normalises is punished for nothing.

    This is not hypothetical: the first bench run against a real plug-in
    reported it losing 1.5 dB where a hand measurement said it gained 1.5.
    """
    x = probes.default_material(RATE, 3.0)
    louder = x * 4.0
    assert metrics.log_spectral_distance(louder, x, RATE) < 0.01
    assert metrics.log_spectral_distance(louder, x, RATE, match_level=False) > 5


def test_suppression_is_positive_when_something_was_removed():
    x = probes.default_material(RATE, 2.0)
    assert metrics.suppression(x, x * 0.1) > 19
    assert abs(metrics.suppression(x, x)) < 1e-6
