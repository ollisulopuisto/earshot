"""The listening page: matched levels, and every take reachable by path."""

import json

import numpy as np
import soundfile as sf

from earshot import listen, probes
from earshot.cli import main

RATE = 48000


def _set(tmp_path):
    speech = probes.default_material(RATE, 2.0)
    folder = tmp_path / "hum"
    folder.mkdir()
    sf.write(folder / "00-ref.wav", speech, RATE)
    sf.write(folder / "01-quieter.wav", speech * 0.5, RATE)
    (folder / "about.json").write_text(json.dumps(
        {"title": "Hurina", "takes": {"01-quieter.wav": {"label": "Hiljaisempi"}}}))
    # A folder with one take is not a comparison.
    (tmp_path / "alone").mkdir()
    sf.write(tmp_path / "alone" / "00.wav", speech, RATE)
    return tmp_path


def test_takes_are_matched_to_the_first(tmp_path):
    data = listen.manifest(_set(tmp_path))
    assert [c["id"] for c in data] == ["hum"]
    ref, quieter = data[0]["takes"]
    assert ref["match_db"] == 0.0
    assert abs(quieter["match_db"] - 20 * np.log10(2)) < 0.1
    assert quieter["label"] == "Hiljaisempi"
    assert quieter["file"] == "hum/01-quieter.wav"


def test_the_page_carries_the_manifest(tmp_path):
    assert main(["listen", str(_set(tmp_path)), "--title", "Koe"]) == 0
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<title>Koe</title>" in page
    assert "hum/01-quieter.wav" in page
    assert "__DATA__" not in page
