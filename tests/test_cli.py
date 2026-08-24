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
