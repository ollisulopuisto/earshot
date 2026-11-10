"""Compare two independent log-spectral distance scores with a shared plan on synthetic audio.

Run: uv run python scripts/benchmark_log_spectral_distance_plan.py > results/2026-11-09-log-spectral-distance-plan.json
This measures execution cost, not restoration quality. Both paths include setup.
"""

import json
import platform
import statistics
import time

import numpy as np

from earshot import degrade, metrics, probes

RATE = 48000
clean = probes.default_material(RATE, 10.0)
damage = next(d for d in degrade.RECIPES if d.name == "narrowband-voip")
broken = np.asarray(damage.apply(clean, RATE))
restored = broken.copy()
low, high = probes._damaged_band(damage, RATE)


def separate():
    return [
        metrics.log_spectral_distance(broken, clean, RATE, low, high),
        metrics.log_spectral_distance(restored, clean, RATE, low, high),
    ]


def shared():
    plan = metrics.LogSpectralDistancePlan(clean, RATE, low, high)
    return [plan.score(broken), plan.score(restored)]


assert separate() == shared()
for _ in range(3):
    separate()
    shared()
times = {"separate": [], "shared": []}
for trial in range(20):
    order = [("separate", separate), ("shared", shared)]
    if trial % 2:
        order.reverse()
    for name, fn in order:
        start = time.perf_counter()
        for _ in range(5):
            fn()
        times[name].append((time.perf_counter() - start) / 5)
print(
    json.dumps(
        {
            "material": "synthetic 10-second 48kHz default_material, narrowband-voip damage",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "trials": 20,
            "repeats_per_trial": 5,
            "seconds_per_pair": times,
            "median_seconds": {k: statistics.median(v) for k, v in times.items()},
            "median_speedup": statistics.median(times["separate"])
            / statistics.median(times["shared"]),
            "score_parity": separate() == shared(),
        },
        indent=2,
    )
)
