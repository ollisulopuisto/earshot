"""Time the probe band-filter design, uncached versus the lru_cache in probes._bands_sos.

Run: uv run python scripts/benchmark_bands_sos.py > results/2026-11-11-bands-sos.json
This measures execution cost only; the filters themselves are unchanged.
"""

import json
import platform
import statistics
import time

import scipy

from earshot import probes

RATE = 48000
ITERATIONS = 500
direct = probes._bands_sos.__wrapped__

# Prime the cache and warm up both paths so the timed calls are not cold.
probes._bands_sos(RATE)
for _ in range(3):
    direct(RATE)
    probes._bands_sos(RATE)

times = {"direct": [], "cached": []}
for trial in range(ITERATIONS):
    order = [("direct", direct), ("cached", probes._bands_sos)]
    if trial % 2:
        order.reverse()
    for name, fn in order:
        start = time.perf_counter()
        fn(RATE)
        times[name].append(time.perf_counter() - start)


def p90(values):
    ordered = sorted(values)
    return ordered[int(0.9 * (len(ordered) - 1))]


print(
    json.dumps(
        {
            "measured": "design of the 4 BANDS Butterworth SOS filters for one sample rate",
            "rate": RATE,
            "iterations": ITERATIONS,
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "median_seconds": {k: statistics.median(v) for k, v in times.items()},
            "p90_seconds": {k: p90(v) for k, v in times.items()},
        },
        indent=2,
    )
)
