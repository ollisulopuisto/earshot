"""Third-party code, copied in and pinned.

Vendoring is a fork unless the drift is handled, so it is handled here:

* ``PINNED`` records what was copied, from where, at which commit, and the
  SHA-256 of the file as it was taken. Results in ``results/`` were measured
  with exactly this code.
* ``tests/test_vendor.py`` checks the file on disk still hashes to the
  recorded value. That catches the dangerous case — a local edit — which
  would otherwise make our numbers silently not-LavaSR any more.
* ``.github/workflows/vendor-check.yml`` fetches upstream on a schedule and
  fails if it has moved. It never updates anything: it tells a human, who
  re-vendors deliberately and re-runs the bench, because new code means new
  numbers and the old ones stop being comparable.

Model weights are not vendored. They are downloaded and checked against a
digest (``earshot.fetch``), so the same pinning applies to them without
putting 58 MB in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pinned:
    """One vendored file and where it came from."""

    path: str
    project: str
    url: str
    commit: str
    taken: str
    sha256: str
    licence: str


PINNED: tuple[Pinned, ...] = (
    Pinned(
        path="lavasr_core.py",
        project="LavaSR-ONNX",
        url="https://github.com/Topping1/LavaSR-ONNX",
        commit="1a979b80d760f00d973b13d530fdd8da51be160b",
        taken="2026-08-24",
        sha256="f09194539d4b80b203ad95d92416f62f7f1675b03aa415603a727d24227c766f",
        licence="Apache-2.0",
    ),
    Pinned(
        path="lavasr_config.yaml",
        project="LavaSR-ONNX",
        url="https://github.com/Topping1/LavaSR-ONNX",
        commit="1a979b80d760f00d973b13d530fdd8da51be160b",
        taken="2026-08-24",
        sha256="5fd2cb08f6cb4b23eb20b6a71e8a4125d2fc6ff74575aecb8c8883fbf161830c",
        licence="Apache-2.0",
    ),
)
