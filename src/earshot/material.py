"""Choosing what to measure on.

A result from twenty seconds of one voice is an anecdote. Restoration
behaves differently on a low voice than a high one, on a close mic than a
distant one, and on a fast talker than a slow one — so the bench takes many
short excerpts across many recordings and reports the spread as well as the
average. An engine that wins on average and loses badly on one speaker is a
different proposition from one that wins everywhere, and only the spread
shows the difference.

Two things this module refuses to do:

**It does not pick at random.** A random window into a podcast is often
silence, a cough, or the tail of a laugh. Excerpts are chosen where there is
speech, by level, and windows that are mostly quiet are rejected. Otherwise
half the measurements are of nothing and the average is meaningless.

**It does not pick twice from the same place.** Excerpts from one recording
are spread across it, because two windows ten seconds apart are the same
voice saying similar things in the same room, and counting them twice makes
a sample look larger than it is.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as audio_io

EXTENSIONS = (".wav", ".wave", ".flac", ".aif", ".aiff", ".w64", ".caf")

# A window is rejected if its level is this far below the recording's own
# speech level. Not an absolute threshold: recordings differ by tens of
# decibels, and a fixed number would take everything from a hot file and
# nothing from a quiet one.
QUIET_MARGIN_DB = 12.0


@dataclass(frozen=True)
class Excerpt:
    """One piece of material, and where it came from."""

    path: Path
    start: float
    seconds: float
    audio: np.ndarray
    rate: int

    @property
    def label(self) -> str:
        return f"{self.path.name} @ {self.start:.0f}s"

    @property
    def speaker(self) -> str:
        """A guess at whose voice this is, from the filename.

        The bench groups by it to report per-speaker spread. A guess is
        enough — it only has to be stable and to separate files that are
        genuinely different people, which a filename usually does because
        that is how anyone names a multitrack recording.
        """
        return self.path.stem.split()[0].lower() if self.path.stem else "unknown"


def recordings(where: str | Path) -> list[Path]:
    """Every readable recording under a path, sorted for reproducibility."""
    root = Path(where)
    if root.is_file():
        return [root]
    found = [
        p
        for p in sorted(root.rglob("*"))
        if p.suffix.lower() in EXTENSIONS and not p.name.startswith(".")
    ]
    # Anything the bench itself wrote is not material.
    return [p for p in found if not p.name.startswith(("00-", "01-", "02-"))]


def _speech_level(x: np.ndarray, rate: int) -> float:
    frame = max(1, int(0.4 * rate))
    count = max(1, len(x) // frame)
    levels = np.array(
        [
            20 * np.log10(np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) + 1e-12)
            for i in range(count)
        ]
    )
    return float(np.percentile(levels, 80))


def excerpts(
    where: str | Path,
    seconds: float = 12.0,
    count: int = 6,
    seed: int = 0,
) -> list[Excerpt]:
    """Pick ``count`` speech-bearing excerpts, spread across the material.

    Deterministic for a given path, length and seed: the same call gives the
    same excerpts, so two engines are compared on identical audio and a
    result can be reproduced by someone else.
    """
    files = recordings(where)
    if not files:
        raise FileNotFoundError(f"no recordings under {where}")

    # Round-robin over files rather than sampling the pool, so one long
    # recording cannot supply every excerpt and drown out the other voices.
    per_file: dict[Path, list[Excerpt]] = {}
    for path in files:
        per_file[path] = _from_one(path, seconds, count, seed)

    picked: list[Excerpt] = []
    index = 0
    while len(picked) < count and any(per_file.values()):
        for path in files:
            queue = per_file.get(path) or []
            if queue:
                picked.append(queue.pop(0))
            if len(picked) >= count:
                break
        index += 1
        if index > count * 2:
            break
    return picked[:count]


def _from_one(path: Path, seconds: float, count: int, seed: int) -> list[Excerpt]:
    """Candidate excerpts from one recording, best first, spread apart."""
    try:
        whole, rate = audio_io.read(path)
    except Exception:
        return []
    window = int(seconds * rate)
    if len(whole) < window:
        return []

    speech = _speech_level(whole, rate)
    # Step through in window-sized hops: candidates never overlap, so two
    # excerpts cannot be the same words.
    starts = list(range(0, len(whole) - window + 1, window))
    if not starts:
        return []

    scored = []
    for start in starts:
        piece = whole[start : start + window]
        level = 20 * np.log10(np.sqrt(np.mean(piece**2)) + 1e-12)
        if level < speech - QUIET_MARGIN_DB:
            continue
        scored.append((level, start, piece))
    if not scored:
        return []

    # Spread the choice over the whole recording rather than taking the
    # loudest run of windows, which would all come from one animated passage.
    # The seed is folded into the path so different files start at different
    # offsets, and the same file always starts at the same one.
    digest = hashlib.sha1(f"{path}|{seed}".encode()).digest()
    offset = int.from_bytes(digest[:4], "big") % max(1, len(scored))
    ordered = scored[offset:] + scored[:offset]
    stride = max(1, len(ordered) // max(1, count))
    chosen = ordered[::stride][:count]
    return [
        Excerpt(path, start / rate, seconds, piece.astype(np.float32), rate)
        for _, start, piece in chosen
    ]
