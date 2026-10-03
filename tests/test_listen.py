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


def test_the_page_starts_blind(tmp_path):
    """Takes are heard before they are named: a label is a bias, and the
    first impressions of the new candidates were taken with names showing
    (2026-10-03). Sokko starts on, under a storage key no earlier page used,
    so a browser that remembered "off" from before starts blind too."""
    assert main(["listen", str(_set(tmp_path))]) == 0
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'id="sw-blind" aria-pressed="true"' in page
    assert "blind: true" in page
    assert 'bindSwitch("sw-blind", "blind", "blind-default-on")' in page


def test_every_take_can_be_picked_as_best(tmp_path):
    assert main(["listen", str(_set(tmp_path))]) == 0
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'pick.className = "pick"' in page
    assert 'fetch("vote"' in page


def _post(port, body: bytes):
    import urllib.error
    import urllib.request

    request = urllib.request.Request(f"http://127.0.0.1:{port}/vote", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_the_server_keeps_each_pick_with_what_was_hidden(tmp_path):
    """A pick is only worth something if it says what was really picked and
    whether its name was showing; the page sends both, the file keeps them."""
    import threading
    import urllib.request

    folder = _set(tmp_path)
    listen.build(folder)
    server = listen.serve(folder, port=0, host="127.0.0.1")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        vote = {"comparison": "hum", "file": "hum/01-quieter.wav", "label": "Hiljaisempi",
                "shown_as": "Otto A", "blind": True}
        assert _post(port, json.dumps(vote).encode()) == 204
        assert _post(port, b"not json") == 400
        assert _post(port, json.dumps({"comparison": "hum"}).encode()) == 400
        assert _post(port, b"{" + b" " * 20000 + b"}") == 413
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html") as page:
            assert page.status == 200
    finally:
        server.shutdown()
    lines = (folder / "votes.jsonl").read_text().splitlines()
    assert len(lines) == 1
    kept = json.loads(lines[0])
    assert kept["file"] == "hum/01-quieter.wav" and kept["blind"] is True
    assert kept["at"].endswith("Z")


def test_votes_summarise_the_latest_pick_per_comparison(tmp_path, capsys):
    (tmp_path / "votes.jsonl").write_text("\n".join(json.dumps(v) for v in [
        {"comparison": "hum", "file": "hum/01-a.wav", "label": "A", "shown_as": "Otto B",
         "blind": True, "at": "2026-10-03T10:00:00Z"},
        {"comparison": "hum", "file": "hum/02-b.wav", "label": "B", "shown_as": "Otto A",
         "blind": True, "at": "2026-10-03T10:05:00Z"},
    ]) + "\n")
    assert main(["votes", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "hum" in out and "B" in out and "blind" in out


def test_the_reference_cannot_be_picked(tmp_path):
    """The reference is the original before damage, best by definition; a
    Paras button on it asked a question with no answer (owner, 2026-10-03)."""
    assert main(["listen", str(_set(tmp_path))]) == 0
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "if (takeIndex !== 0) row.append(pick)" in page
