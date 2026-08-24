"""Has upstream moved since we vendored?

Run by `.github/workflows/vendor-check.yml`. Exits non-zero when a pinned
file differs from what upstream has now, so that a human decides what to do.
It deliberately cannot update anything: a vendor bump has to arrive with new
bench results, and only a person can run those.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request

from earshot import vendor

RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def upstream(pin: vendor.Pinned, ref: str = "HEAD") -> str | None:
    repo = pin.url.removeprefix("https://github.com/")
    # The file may be named differently upstream than in our tree.
    name = pin.path.replace("lavasr_config.yaml", "config.yaml")
    url = RAW.format(repo=repo, ref=ref, path=name)
    try:
        with urllib.request.urlopen(url) as response:
            return hashlib.sha256(response.read()).hexdigest()
    except Exception as exc:  # noqa: BLE001 - the report is the point
        print(f"  could not fetch {url}: {exc}")
        return None


def main() -> int:
    moved = []
    for pin in vendor.PINNED:
        now = upstream(pin)
        if now is None:
            print(f"? {pin.path}: upstream unreachable, not treating as drift")
            continue
        if now == pin.sha256:
            print(f"ok {pin.path}: unchanged since {pin.commit[:7]}")
        else:
            moved.append(pin)
            print(f"!! {pin.path}: upstream has changed")
            print(f"     pinned {pin.sha256}")
            print(f"     now    {now}")

    if not moved:
        return 0
    print(
        "\nUpstream has moved. Do not just copy it in:\n"
        "  1. read what changed and decide whether it matters here\n"
        "  2. re-vendor and update PINNED\n"
        "  3. re-run the bench and commit new results in the same change\n"
        "The stored numbers are about the pinned code, and stop being\n"
        "comparable the moment it changes."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
