"""Vendored code must be what it claims to be.

The failure this prevents is quiet: someone tweaks a vendored file to fix a
bug, and every number in `results/` is now about a modified version of a
project that is named, unmodified, in the table.
"""

import hashlib
from pathlib import Path

import pytest

from earshot import vendor

HERE = Path(vendor.__file__).parent


@pytest.mark.parametrize("pin", vendor.PINNED, ids=lambda p: p.path)
def test_the_file_is_what_was_pinned(pin):
    path = HERE / pin.path
    assert path.exists(), f"{pin.path} is recorded in PINNED but not present"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == pin.sha256, (
        f"{pin.path} has been edited since it was vendored from {pin.project}.\n"
        "Either restore it, or re-vendor deliberately and re-run the bench — "
        "changed code means the stored results are about different code."
    )


def test_every_vendored_file_is_accounted_for():
    """A file nobody recorded is a file nobody can trace."""
    recorded = {p.path for p in vendor.PINNED}
    present = {
        p.name
        for p in HERE.iterdir()
        if p.is_file() and p.name not in {"__init__.py", "README.md"}
        and not p.name.startswith("LICENSE")
    }
    assert present == recorded, f"not recorded in PINNED: {present - recorded}"


def test_each_pin_names_its_licence_and_origin():
    for pin in vendor.PINNED:
        assert pin.licence and pin.url and pin.commit
        assert (HERE / f"LICENSE-{pin.project}").exists(), pin.project
