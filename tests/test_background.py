"""Real background noise under the voice, for the goal stated 2026-10-04:
only voices are wanted, everything else is to go.

The recipes until now had only synthetic hiss and mains hum. DEMAND (Thiemann,
Ito, Vincent 2013, CC BY 4.0) is recorded noise at 48 kHz: an office, a
kitchen, a cafeteria with other people's voices in it. The damage must keep
the sample count, hit its stated SNR against the speech, and be the same
every time.
"""

import numpy as np
import pytest

from earshot import degrade, engines, probes

RATE = 48000


def _clean():
    return probes.default_material(RATE, 6.0)


def _apply(name):
    try:
        return degrade.by_name(name).apply(_clean(), RATE)
    except (engines.EngineError, RuntimeError) as exc:
        pytest.skip(f"DEMAND unavailable here: {exc}")


def test_every_environment_is_pinned():
    assert set(degrade.ENVIRONMENTS) == {"OOFFICE", "DKITCHEN", "PCAFETER"}
    for asset in degrade.ENVIRONMENTS.values():
        assert len(asset.sha256) == 64 and asset.url.startswith("https://zenodo.org/")


@pytest.mark.parametrize("name, snr", [("office", 15.0), ("kitchen", 10.0), ("cafeteria", 10.0)])
def test_the_background_sits_at_its_snr(name, snr):
    clean = _clean()
    damaged = _apply(name)
    assert len(damaged) == len(clean)
    added = damaged - clean
    measured = 20 * np.log10(degrade._speech_rms(clean, RATE) / np.sqrt(np.mean(added**2)))
    assert measured == pytest.approx(snr, abs=0.5)


def test_it_is_the_same_every_time():
    np.testing.assert_array_equal(_apply("kitchen"), _apply("kitchen"))


def test_the_recipes_say_they_need_the_recordings():
    for name in ("office", "kitchen", "cafeteria"):
        assert degrade.by_name(name).needs == "demand"


def test_without_downloads_or_cache_the_bench_is_told(tmp_path, monkeypatch):
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path))
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    assert not probes._have("demand")
