"""Is the restored voice still its owner's?

UniPASE won the owner's blind picks 7 of 8 and keeps no sample of its input:
it re-speaks everything. `origin` reads near zero for it by design and
cannot say whether the voice is the same person's. Speaker similarity can:
the cosine between speaker embeddings of the clean original and the take.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from earshot import degrade, engines, metrics, probes, speaker

RATE = 48000
EARS = Path(__file__).parent.parent / "out/colab-calls/out/kuuntelu-bwe-ears"


def _available():
    try:
        speaker.embed(np.zeros(RATE, dtype=np.float32) + 1e-3, RATE)
    except engines.EngineError as exc:
        pytest.skip(f"speaker model unavailable here: {exc}")


def test_the_weights_are_pinned():
    (asset,) = speaker.ASSETS
    assert len(asset.sha256) == 64 and "/resolve/main/" not in asset.url


def test_a_voice_is_itself_and_a_gain_change_is_still_itself():
    _available()
    x = probes.default_material(RATE, 4.0)
    assert metrics.speaker_similarity(x, x, RATE) > 0.999
    assert metrics.speaker_similarity(x, 0.25 * x, RATE) > 0.99


def test_two_people_are_two_people():
    _available()
    files = sorted(EARS.glob("*/00-alkuperainen.wav")) if EARS.exists() else []
    if len(files) < 2:
        pytest.skip("needs two EARS excerpts from different speakers in out/")
    a, _ = sf.read(EARS / "01-puhdas/00-alkuperainen.wav", dtype="float32")  # p008
    b, _ = sf.read(EARS / "02-puhelin-p001/00-alkuperainen.wav", dtype="float32")  # p001
    same = metrics.speaker_similarity(a[: RATE * 4], a[RATE * 4:], RATE)
    other = metrics.speaker_similarity(a, b, RATE)
    assert same > other + 0.2


def test_passthrough_changes_no_ones_voice():
    _available()
    run = probes.run_all(engines.load("passthrough"), probes.default_material(RATE, 6.0),
                         RATE, degrade.by_name("narrowband-voip"))
    assert run.value("speaker", "after") == pytest.approx(run.value("speaker", "before"), abs=1e-4)
    assert abs(run.value("speaker", "change")) < 1e-4
