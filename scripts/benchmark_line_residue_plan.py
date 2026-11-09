"""Compare two independent line scores with a shared plan on synthetic audio.

Run: uv run python scripts/benchmark_line_residue_plan.py > results/line-residue-plan.json
This measures execution cost, not restoration quality. Both paths include setup.
"""

import json
import platform
import statistics
import time

import numpy as np

from earshot import metrics, probes

RATE = 48000
clean = probes.default_material(RATE, 10.0)
rng = np.random.default_rng(15)
errors = [rng.normal(0, 0.01, len(clean)), np.zeros_like(clean)]


def separate():
    return [metrics.line_residue(error, clean, RATE) for error in errors]


def shared():
    plan = metrics.LineResiduePlan(clean, RATE)
    return [plan.score(error) for error in errors]


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
            "material": "synthetic 10-second 48kHz default_material, noise and zero errors",
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
