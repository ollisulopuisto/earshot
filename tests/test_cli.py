"""The command has to work on a machine with nothing installed."""

from earshot import cli


def test_bench_runs_with_no_material_and_no_engines(capsys):
    assert cli.main(["bench", "--seconds", "3", "--damage", "hiss"]) == 0
    out = capsys.readouterr().out
    assert "earshot bench" in out
    assert "synthetic material" in out
    assert "passthrough" in out


def test_damages_lists_the_recipes(capsys):
    assert cli.main(["damages"]) == 0
    out = capsys.readouterr().out
    assert "narrowband-voip" in out and "needs ffmpeg" in out


def test_report_is_written_to_a_directory(tmp_path, capsys):
    assert cli.main(["bench", "--seconds", "3", "--damage", "clean",
                     "--out", str(tmp_path), "--write"]) == 0
    report = tmp_path / "report.md"
    assert report.exists() and "origin" in report.read_text()
    assert (tmp_path / "clean" / "00-clean.wav").exists()


def test_an_unloadable_engine_fails_loudly(capsys):
    assert cli.main(["bench", "--engine", "vst3:/ei/ole.vst3", "--seconds", "2"]) == 2
    assert "no plug-in at" in capsys.readouterr().err


def test_scoreboard_reads_stored_results(tmp_path, capsys):
    """Results accumulate over months and machines; the summary is generated
    from them so it cannot drift from the numbers it claims to summarise."""
    assert cli.main(["bench", "--seconds", "3", "--damage", "hiss",
                     "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli.main(["scoreboard", str(tmp_path / "results.json")]) == 0
    out = capsys.readouterr().out
    assert "passthrough" in out and "hiss" in out and "recovered" in out


def test_scoreboard_says_so_when_there_is_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["scoreboard"]) == 1
    assert "no results found" in capsys.readouterr().err


def test_per_speaker_tables_name_the_losing_voice(tmp_path, capsys):
    """A mean plus a spread says an engine is inconsistent; this says on whom."""
    import numpy as np
    import soundfile as sf

    from earshot import probes

    rate = 48000
    for name in ("alpha take.wav", "beta take.wav"):
        sf.write(tmp_path / name, probes.default_material(rate, 20.0), rate)

    assert cli.main(["bench", "--input", str(tmp_path), "--seconds", "4",
                     "--excerpts", "4", "--damage", "hiss",
                     "--engine", "passthrough:3"]) == 0
    out = capsys.readouterr().out
    assert "per speaker" in out
    assert "alpha" in out and "beta" in out
