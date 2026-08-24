"""Getting model weights onto the machine, once.

Every engine that wraps an open model needs the same thing: a file that is
too big to commit, fetched from somewhere, kept between runs. Doing that
once here means engine authors write a declaration instead of a download,
and it means there is one place where the rules live.

Three rules, and each is a thing that has gone wrong in somebody's project:

**A download is never silent.** It says what it is fetching and how big,
on stderr, before it starts. A tool that quietly pulls 300 MB the first time
you run it has spent someone's tethered connection without asking.

**The checksum is not optional.** A model file that changed upstream turns
every number this bench has ever produced into a number about a different
model, and nothing else in the system would notice. The digest is recorded
in the repo next to the URL; a mismatch deletes the file and raises.

**It can be switched off.** ``EARSHOT_NO_DOWNLOAD=1`` makes fetching raise
instead of reaching the network, with the URL and the destination in the
message so it can be done by hand. CI sets it, and so can anyone who would
rather see what is being asked for first.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .engines import EngineError

CHUNK = 1 << 20


@dataclass(frozen=True)
class Asset:
    """One file an engine needs, and how to know it is the right one."""

    name: str
    url: str
    sha256: str
    about: str = ""

    @property
    def path(self) -> Path:
        return cache_dir() / self.name


def cache_dir() -> Path:
    """Where downloaded weights live. Safe to delete; costs one refetch."""
    root = Path(
        os.environ.get("EARSHOT_CACHE")
        or os.environ.get("XDG_CACHE_HOME")
        or (Path.home() / ".cache")
    )
    if not os.environ.get("EARSHOT_CACHE"):
        root = root / "earshot"
    root.mkdir(parents=True, exist_ok=True)
    return root


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


def ensure(asset: Asset) -> Path:
    """Return the local path to ``asset``, downloading it if need be.

    A file already present is verified, not trusted: a half-written download
    from an interrupted run is exactly the case where the bytes are there
    and wrong.
    """
    target = asset.path
    # A name may carry a subdirectory: ONNX graphs reference their external
    # weight files by the exact name they were built with, so those files
    # have to keep it and sit together rather than be renamed into one flat
    # cache.
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) == asset.sha256:
            return target
        # Wrong contents. Say so and refetch rather than leaving a file that
        # will silently produce numbers about a different model.
        print(
            f"earshot: {asset.name} does not match its checksum, refetching",
            file=sys.stderr,
        )
        target.unlink()

    if os.environ.get("EARSHOT_NO_DOWNLOAD"):
        raise EngineError(
            f"{asset.name} is not present and downloading is disabled "
            f"(EARSHOT_NO_DOWNLOAD). Fetch it by hand:\n"
            f"  curl -L -o {target} {asset.url}"
        )

    note = f" — {asset.about}" if asset.about else ""
    print(f"earshot: fetching {asset.name}{note}\n  from {asset.url}", file=sys.stderr)
    part = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(asset.url) as response, open(part, "wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise EngineError(f"could not fetch {asset.name}: {exc}") from exc

    got = digest(part)
    if got != asset.sha256:
        part.unlink(missing_ok=True)
        raise EngineError(
            f"{asset.name} downloaded but the checksum is wrong.\n"
            f"  expected {asset.sha256}\n  got      {got}\n"
            "Either the file upstream changed — in which case every number "
            "measured with it is about a different model — or the download "
            "was corrupted. Do not update the digest without checking which."
        )
    part.replace(target)
    return target


def missing(assets) -> list[Asset]:
    """Which of these are absent or fail their digest."""
    out = []
    for asset in assets:
        if not asset.path.exists() or digest(asset.path) != asset.sha256:
            out.append(asset)
    return out


def ensure_all(assets) -> list[Path]:
    """Fetch every asset, reporting all of them at once when it cannot.

    One at a time would be one round trip per file: someone with downloads
    disabled and five models to fetch should get five commands, not the
    first one and then another error after they run it.
    """
    assets = list(assets)
    if os.environ.get("EARSHOT_NO_DOWNLOAD"):
        absent = missing(assets)
        if absent:
            lines = "\n".join(f"  curl -L --create-dirs -o {a.path} {a.url}"
                               for a in absent)
            raise EngineError(
                f"{len(absent)} file(s) missing and downloading is disabled "
                f"(EARSHOT_NO_DOWNLOAD). Fetch them by hand:\n{lines}"
            )
    return [ensure(asset) for asset in assets]
