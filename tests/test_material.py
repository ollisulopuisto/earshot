"""Excerpts must be speech, spread out, and the same every time.

A random window into a podcast is often silence. Half the measurements
would then be of nothing, and the average would mean nothing.
"""

import numpy as np
import pytest
import soundfile as sf

from earshot import material, probes

RATE = 48000


@pytest.fixture
def corpus(tmp_path):
    """Two 'speakers', each with a long stretch of silence in the middle."""
    for name in ("nyman take.wav", "wancke take.wav"):
        speech = probes.default_material(RATE, 30.0)
        speech[int(10 * RATE) : int(20 * RATE)] = 0.0
        sf.write(tmp_path / name, speech, RATE)
    (tmp_path / "notes.txt").write_text("not audio")
    return tmp_path


def test_only_recordings_are_found(corpus):
    found = material.recordings(corpus)
    assert len(found) == 2
    assert all(p.suffix == ".wav" for p in found)


def test_excerpts_avoid_the_silence(corpus):
    picked = material.excerpts(corpus, seconds=4.0, count=6)
    assert len(picked) == 6
    for piece in picked:
        level = 20 * np.log10(np.sqrt(np.mean(piece.audio**2)) + 1e-12)
        assert level > -60, f"{piece.label} is silence"


def test_both_speakers_are_represented(corpus):
    picked = material.excerpts(corpus, seconds=4.0, count=4)
    assert {p.speaker for p in picked} == {"nyman", "wancke"}


def test_the_same_call_gives_the_same_excerpts(corpus):
    a = material.excerpts(corpus, seconds=4.0, count=4)
    b = material.excerpts(corpus, seconds=4.0, count=4)
    assert [(p.path, p.start) for p in a] == [(p.path, p.start) for p in b]


def test_excerpts_do_not_overlap_within_a_file(corpus):
    picked = material.excerpts(corpus, seconds=4.0, count=6)
    for path in {p.path for p in picked}:
        starts = sorted(p.start for p in picked if p.path == path)
        assert all(b - a >= 4.0 for a, b in zip(starts, starts[1:]))


def test_a_single_file_still_works(corpus):
    one = next(iter(material.recordings(corpus)))
    picked = material.excerpts(one, seconds=4.0, count=3)
    assert len(picked) == 3 and {p.path for p in picked} == {one}


def test_an_empty_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        material.excerpts(tmp_path)



def test_recordings_keeps_a_bracketed_source_with_no_base(tmp_path):
    """The restore-output rule needs the base file to exist: a source that
    happens to carry brackets in its name is still material."""
    import soundfile as sf

    from earshot import material, probes

    rate = 48000
    sf.write(tmp_path / "take [raw].wav", probes.default_material(rate, 1.0), rate)
    assert [p.name for p in material.recordings(tmp_path)] == ["take [raw].wav"]



def test_recordings_keeps_a_bracketed_source_in_another_directory(tmp_path):
    """The restore-output rule is per-folder: folderB/take.wav does not make
    folderA/take [raw].wav anyone's output."""
    import soundfile as sf

    from earshot import material, probes

    rate = 48000
    (tmp_path / "folderA").mkdir()
    (tmp_path / "folderB").mkdir()
    sf.write(tmp_path / "folderA" / "take [raw].wav",
             probes.default_material(rate, 1.0), rate)
    sf.write(tmp_path / "folderB" / "take.wav",
             probes.default_material(rate, 1.0), rate)
    assert [p.name for p in material.recordings(tmp_path)] == [
        "take [raw].wav", "take.wav",
    ]
