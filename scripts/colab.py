"""Run an earshot command on a Colab GPU and bring the result back.

    uv run python scripts/colab.py --dry-run
    uv run python scripts/colab.py              # the bwe-ears listening set
    uv run python scripts/colab.py --run "uv run earshot bench ..." --result results

UniverSR takes about 12 minutes per 8 s on an M1 Max CPU; a GPU is the
difference between a bench run being an overnight job and a coffee.

The pattern is the podcast repo's colab-transcribe: `plan` is the only place
`colab` commands are built, and `--dry-run` prints exactly what a real run
executes. Needs the `colab` CLI (google-colab-cli) and its login.

**What leaves this machine.** By default nothing but the commit hash and the
command. The VM clones the repository from GitHub and fetches EARS from its
public release. The owner's own recordings go up only when named with
``--upload``, and listing them there is the decision to send them to Google.

**What runs.** The exact commit checked out here, which must already be on
GitHub. Not the branch name: a branch can move between this check and the
clone, and then the VM measures code nobody tested here.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from colab_job import RESULT_TAR, UPLOAD_TAR  # noqa: E402

JOB = ROOT / "scripts" / "colab_job.py"
# No deepfilternet: deepfilterlib has no wheel for Colab's Python 3.13 and
# needs a Rust build there (first run, 2026-10-02). Ask for it explicitly.
DEFAULT_EXTRAS = "universr,unipase,novasr,lavasr"
# Six hours: past any bench run here, and short of leaving a GPU billed
# overnight if something hangs.
TIMEOUT_S = 6 * 3600


@dataclass(frozen=True)
class Options:
    session: str
    gpu: str
    commit: str
    run: str
    result: str
    out: Path
    extras: tuple[str, ...] = ()
    ears: tuple[str, ...] = ()
    upload: tuple[Path, ...] = field(default_factory=tuple)
    keep: bool = False
    audiosr: bool = False


# The tool's default login asks for a code pasted from a browser, which an
# unattended run cannot give. This machine authenticates with Google ADC.
COLAB = ["colab", "--auth", "adc"]


def plan(o: Options) -> list[list[str]]:
    """Every command of the run, first to last."""
    local_upload = str(Path(tempfile.gettempdir()) / f"earshot-{o.session}-upload.tar")
    local_result = str(Path(tempfile.gettempdir()) / f"earshot-{o.session}-result.tar")
    commands = [[*COLAB, "new", "-s", o.session, "--gpu", o.gpu]]
    if o.upload:
        commands += [
            ["tar", "-cf", local_upload, "-C", str(ROOT), *map(str, o.upload)],
            [*COLAB, "upload", "-s", o.session, local_upload, UPLOAD_TAR],
        ]
    env = {
        "EARSHOT_COMMIT": o.commit,
        "EARSHOT_EXTRAS": ",".join(o.extras),
        "EARSHOT_EARS": ",".join(o.ears),
        "EARSHOT_RUN": o.run,
        "EARSHOT_RESULT": o.result,
        "EARSHOT_AUDIOSR": "1" if o.audiosr else "",
    }
    job = [*COLAB, "exec", "-s", o.session, "-f", str(JOB), "--timeout", str(TIMEOUT_S)]
    for key, value in env.items():
        job += ["--env", f"{key}={value}"]
    commands += [
        job,
        [*COLAB, "download", "-s", o.session, RESULT_TAR, local_result],
        ["mkdir", "-p", str(o.out)],
        ["tar", "-xf", local_result, "-C", str(o.out)],
    ]
    if not o.keep:
        commands.append([*COLAB, "stop", "-s", o.session])
    return commands


def _on_origin(sha: str) -> bool:
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT, check=False)
    remote = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], cwd=ROOT, capture_output=True, text=True
    )
    return remote.returncode == 0 and bool(remote.stdout.strip())


def check_pushed(sha: str) -> None:
    if not _on_origin(sha):
        raise SystemExit(
            f"{sha[:7]} is not on GitHub: push it first. The VM clones from there "
            "and runs exactly this commit."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", default="earshot")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--extras", default=DEFAULT_EXTRAS)
    parser.add_argument("--ears", default="p001,p002,p008",
                        help="EARS speakers fetched on the VM; empty for none")
    parser.add_argument(
        "--run",
        default="uv run python scripts/listening_set.py --set bwe-ears --one-at-a-time "
        "material/local/ears out/kuuntelu-bwe-ears",
    )
    parser.add_argument("--result", default="out/kuuntelu-bwe-ears")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "colab")
    parser.add_argument("--upload", type=Path, nargs="*", default=[],
                        help="repository-relative paths to send; private audio only by choice")
    parser.add_argument("--keep", action="store_true", help="leave the VM running")
    parser.add_argument("--audiosr", action="store_true",
                        help="build AudioSR's own Python 3.10 environment on the VM")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    options = Options(
        session=args.session, gpu=args.gpu, commit=sha, run=args.run, result=args.result,
        out=args.out, extras=tuple(e for e in args.extras.split(",") if e),
        ears=tuple(s for s in args.ears.split(",") if s), upload=tuple(args.upload),
        keep=args.keep, audiosr=args.audiosr,
    )
    commands = plan(options)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0

    check_pushed(sha)
    status = execute(commands, session=args.session, keep=args.keep)
    if not status:
        print(f"result in {args.out / args.result}", file=sys.stderr)
    return status


def execute(commands: list[list[str]], session: str, keep: bool) -> int:
    """Run the plan in order; on a failure, stop the VM unless told to keep it.

    A VM left running holds the free tier's one session, and after
    ``--upload`` it also holds the owner's audio.
    """
    for command in commands:
        print(f"$ {shlex.join(command)}", file=sys.stderr, flush=True)
        if subprocess.run(command, cwd=ROOT).returncode:
            print(f"failed: {shlex.join(command)}", file=sys.stderr)
            if not keep:
                subprocess.run([*COLAB, "stop", "-s", session], cwd=ROOT)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
