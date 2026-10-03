"""The half of a Colab run that executes on the VM.

Sent by `scripts/colab.py` with `colab exec -f`, which runs it in the VM's
own Python kernel, so it imports nothing from earshot until earshot is
installed. Settings arrive as environment variables:

    EARSHOT_COMMIT   the exact commit to check out; the run refuses any other
    EARSHOT_EXTRAS   comma-separated extras to install
    EARSHOT_EARS     comma-separated EARS speakers to fetch, e.g. p001,p008
    EARSHOT_RUN      the shell command to run in the repository
    EARSHOT_RESULT   the path, relative to the repository, to bring back
    EARSHOT_AUDIOSR  "1" to build AudioSR's own Python 3.10 environment

EARS is fetched here from its public GitHub release, so no material leaves
the owner's machine for it. Anything uploaded arrives as
/content/upload.tar and is unpacked into the repository.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO_URL = "https://github.com/ollisulopuisto/earshot"
CHECKOUT = Path("/content/earshot")
UPLOAD_TAR = "/content/upload.tar"
RESULT_TAR = "/content/result.tar"
# The second T4 run died of fragmentation: 3.44 GiB reserved by PyTorch but
# unallocated when 2.2 GiB was asked for.
GPU_ENV = {"EARSHOT_DEVICE": "cuda", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
EARS_RELEASE = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset"


def freeform_name(found: str) -> str | None:
    """The name the listening plans use for an EARS freeform monologue.

    ``p008/freeform_speech_01.wav`` → ``p008 freeform 01.wav``; anything
    else is not a freeform file and keeps its place in ``ears-sent``.
    """
    match = re.search(r"(p\d{3})/freeform\D*(\d+)\.wav$", found)
    return f"{match.group(1)} freeform {match.group(2)}.wav" if match else None


def _sh(command: str, cwd: Path | None = None) -> None:
    # Relayed line by line: in a Jupyter kernel a child's own stdout goes to
    # the kernel's log, not back to `colab exec`, and the run would be silent.
    print(f"$ {command}", flush=True)
    process = subprocess.Popen(
        ["bash", "-lc", command], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise SystemExit(f"failed ({process.returncode}): {command}")


def _fetch_ears(speakers: list[str]) -> None:
    sent = CHECKOUT / "material/local/ears-sent"
    flat = CHECKOUT / "material/local/ears"
    sent.mkdir(parents=True, exist_ok=True)
    flat.mkdir(parents=True, exist_ok=True)
    for speaker in speakers:
        archive = Path(f"/content/{speaker}.zip")
        if not archive.exists():
            _sh(f"curl -sSL --fail -o {archive} {EARS_RELEASE}/{speaker}.zip")
        _sh(f"unzip -q -o {archive} -d {sent}")
    renamed = 0
    for path in sorted(sent.rglob("*.wav")):
        name = freeform_name(path.relative_to(sent).as_posix())
        if name:
            shutil.copyfile(path, flat / name)
            renamed += 1
    if speakers and not renamed:
        raise SystemExit(
            "EARS unpacked but no freeform_*.wav found: the release layout is not "
            "what freeform_name expects, and the listening plans would not find "
            "their files"
        )
    print(f"EARS: {len(speakers)} speakers, {renamed} freeform files", flush=True)


AUDIOSR_ENV = Path("/content/audiosr-env")


def _build_audiosr() -> None:
    """AudioSR pins numpy<=1.23.5, librosa 0.9.2 and transformers 4.30.2, so
    it gets an interpreter of its own; the engine talks to it over a pipe."""
    _sh(f"uv venv -q --python 3.10 {AUDIOSR_ENV}")
    # Each pin is a failure seen on a real run (2026-10-03): librosa 0.9.2
    # imports pkg_resources, which a uv environment lacks and setuptools 81
    # removed; audiosr imports matplotlib without declaring it; torchaudio
    # from 2.9 reads files only through torchcodec. torch 2.4.1 is of
    # AudioSR's era and has CUDA wheels on PyPI.
    _sh(f"uv pip install -q --python {AUDIOSR_ENV}/bin/python audiosr==0.0.7 soundfile "
        "matplotlib 'setuptools<81' torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1")
    os.environ["EARSHOT_AUDIOSR_PYTHON"] = str(AUDIOSR_ENV / "bin/python")


def clear_result() -> None:
    """`colab exec` reports success even when this script fails (the kernel
    swallows the exit), so the result file is the proof a run finished. One
    left by an earlier run on the same VM must not pass for this one's."""
    Path(RESULT_TAR).unlink(missing_ok=True)


def main() -> None:
    commit = os.environ["EARSHOT_COMMIT"]
    extras = [e for e in os.environ.get("EARSHOT_EXTRAS", "").split(",") if e]
    speakers = [s for s in os.environ.get("EARSHOT_EARS", "").split(",") if s]
    run = os.environ["EARSHOT_RUN"]
    result = os.environ["EARSHOT_RESULT"]
    started = time.time()
    clear_result()

    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true")
    if not CHECKOUT.exists():
        _sh(f"git clone -q {REPO_URL} {CHECKOUT}")
    _sh(f"git fetch -q origin && git checkout -q {commit}", cwd=CHECKOUT)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CHECKOUT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != commit:
        raise SystemExit(f"checked out {head}, asked for {commit}")

    flags = " ".join(f"--extra {e}" for e in ["dev", *extras])
    _sh(f"pip install -q uv && uv sync -q {flags}", cwd=CHECKOUT)
    if Path(UPLOAD_TAR).exists():
        _sh(f"tar -xf {UPLOAD_TAR} -C {CHECKOUT}")
    _fetch_ears(speakers)
    if os.environ.get("EARSHOT_AUDIOSR") == "1":
        _build_audiosr()

    for key, value in GPU_ENV.items():
        os.environ.setdefault(key, value)
    _sh(run, cwd=CHECKOUT)
    _sh(f"tar -cf {RESULT_TAR} {result}", cwd=CHECKOUT)
    print(f"done in {time.time() - started:.0f} s, result in {RESULT_TAR}", flush=True)


if __name__ == "__main__" and os.environ.get("EARSHOT_COMMIT"):
    main()
