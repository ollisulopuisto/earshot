"""The damage must be damage, and nothing else.

Every recipe has to keep the sample count and the alignment, because the
bench compares sample against sample. A degradation that shifted the signal
would be scored as damage no engine could ever undo, and the whole bench
would quietly measure the wrong thing.
"""

import numpy as np
import pytest

from earshot import degrade, probes

RATE = 48000


@pytest.fixture(scope="module")
def clean():
    return probes.default_material(RATE, 4.0)


@pytest.mark.parametrize("recipe", [r for r in degrade.RECIPES if not r.needs],
                         ids=lambda r: r.name)
def test_damage_keeps_the_sample_count(recipe, clean):
    out = recipe.apply(clean, RATE)
    assert len(out) == len(clean), recipe.name
    assert np.all(np.isfinite(out)), recipe.name


def test_band_limit_removes_the_band_it_says():
    from scipy import signal
    x = probes.default_material(RATE, 3.0)
    y = degrade.band_limit(x, RATE, low=0, high=4000)
    f, px = signal.welch(x, RATE, nperseg=4096)
    _, py = signal.welch(y, RATE, nperseg=4096)
    above = (f > 6000) & (f < 16000)
    below = (f > 500) & (f < 3000)
    assert 10 * np.log10(py[above].mean() / px[above].mean()) < -30
    assert abs(10 * np.log10(py[below].mean() / px[below].mean())) < 3


def test_clean_is_actually_untouched(clean):
    assert np.array_equal(degrade.by_name("clean").apply(clean, RATE), clean)


def test_noise_hits_the_requested_snr(clean):
    noisy = degrade.noise(clean, RATE, snr_db=20.0)
    added = noisy - clean
    speech = degrade._speech_rms(clean, RATE)
    snr = 20 * np.log10(speech / (np.sqrt(np.mean(added**2)) + 1e-12))
    assert 18 < snr < 22


def test_autogain_flattens_the_dynamics(clean):
    from earshot import metrics
    before = metrics.dynamics(clean, RATE)
    after = metrics.dynamics(degrade.autogain(clean, RATE), RATE)
    # This is the point of the recipe: it destroys the distance between
    # speech and silence, which is the information no engine can get back.
    assert after.range_db < before.range_db - 5


def test_unknown_recipe_names_the_known_ones():
    with pytest.raises(KeyError) as caught:
        degrade.by_name("ei-ole")
    assert "narrowband-voip" in str(caught.value)
