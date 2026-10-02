"""The Colab driver: what it will run, before it runs anything.

`plan` is the only place `colab` commands are built, and `--dry-run` prints
exactly its output — the pattern from the podcast repo's colab-transcribe.
These tests hold the two properties that matter: nothing of the owner's
leaves this machine unless asked for by name, and the VM runs the exact
commit that was checked here, not whatever a branch points at by then.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


colab = _load("colab")
job = _load("colab_job")

SHA = "0123456789abcdef0123456789abcdef01234567"


def _plan(**overrides):
    options = dict(
        session="earshot", gpu="T4", commit=SHA, run="uv run earshot bench",
        result="out/colab", out=Path("/tmp/x"), extras=("universr",),
        ears=("p001",), upload=(), keep=False,
    )
    options.update(overrides)
    return colab.plan(colab.Options(**options))


def _flat(commands):
    return [" ".join(map(str, c)) for c in commands]


def test_a_vm_is_made_first_and_stopped_last():
    commands = _plan()
    assert commands[0][:2] == ["colab", "new"] and "T4" in commands[0]
    assert commands[-1][:2] == ["colab", "stop"]


def test_keep_leaves_the_vm_running():
    assert all(c[:2] != ["colab", "stop"] for c in _plan(keep=True))


def test_nothing_is_uploaded_unless_named():
    """The default material is EARS, fetched on the VM from its public
    release. The owner's voices go up only when listed with --upload."""
    assert not any(c[:2] == ["colab", "upload"] for c in _plan())
    uploading = _plan(upload=(Path("material/local/x.wav"),))
    assert any(c[:2] == ["colab", "upload"] for c in uploading)


def test_the_vm_is_told_the_commit_not_the_branch():
    joined = " ".join(_flat(_plan()))
    assert f"EARSHOT_COMMIT={SHA}" in joined


def test_the_job_gets_its_settings_and_a_long_timeout():
    (job_call,) = [c for c in _plan() if c[:2] == ["colab", "exec"]]
    assert job_call[job_call.index("-f") + 1].endswith("colab_job.py")
    assert float(job_call[job_call.index("--timeout") + 1]) >= 3600
    envs = {job_call[i + 1].split("=", 1)[0] for i, a in enumerate(job_call) if a == "--env"}
    assert {"EARSHOT_COMMIT", "EARSHOT_RUN", "EARSHOT_RESULT", "EARSHOT_EXTRAS",
            "EARSHOT_EARS"} <= envs


def test_results_come_back_as_one_file():
    downloads = [c for c in _plan() if c[:2] == ["colab", "download"]]
    assert len(downloads) == 1
    assert downloads[0][-2] == job.RESULT_TAR


def test_an_unpushed_commit_is_refused(monkeypatch):
    """The VM clones from GitHub. A commit that is not there would fail on
    the VM after a GPU was paid for, or — worse — a branch name would run
    something other than what was tested here."""
    monkeypatch.setattr(colab, "_on_origin", lambda sha: False)
    with pytest.raises(SystemExit) as caught:
        colab.check_pushed(SHA)
    assert "push" in str(caught.value)


@pytest.mark.parametrize(
    "found, expected",
    [
        ("p008/freeform_speech_01.wav", "p008 freeform 01.wav"),
        ("p001/freeform_speech_02.wav", "p001 freeform 02.wav"),
        ("p008/rainbow_03_regular.wav", None),
    ],
)
def test_ears_files_get_the_names_the_listening_plan_uses(found, expected):
    assert job.freeform_name(found) == expected
