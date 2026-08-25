"""``realtime`` has to be a property of the engine, not of when it was timed.

Measured on this machine with LavaSR on one fixed 10 s buffer: cold calls ran
6.9x to 15x realtime and warmed ones 23x to 50x, so a single untimed-first
call reported whatever the CPU happened to be doing. Across three bench runs
the same engine on the same machine read 6-13x, 10-23x and 26-46x. Pinning
the ONNX thread count did not remove it (spread stayed 1.5x to 2.7x at every
setting); an untimed warm-up call did.
"""

import numpy as np

from earshot import degrade, probes
from earshot.engines import Loaded


class SlowFirstCall:
    """An engine that is slow until it has been called once.

    Every real engine behaves this way to some degree — allocation, graph
    optimisation, the CPU deciding this is real work. This makes it explicit
    so the effect can be tested without depending on the machine's mood.
    """

    name = "slow-first-call"

    def __init__(self, cold_iters=4_000_000, warm_iters=100_000):
        self.calls = 0
        self.cold_iters = cold_iters
        self.warm_iters = warm_iters

    def process(self, audio, rate):
        self.calls += 1
        spin = self.cold_iters if self.calls == 1 else self.warm_iters
        total = 0.0
        for i in range(spin):          # burn CPU, deterministically
            total += i
        assert total >= 0.0
        return np.asarray(audio, dtype=np.float32)


def test_the_cold_call_does_not_set_the_reported_speed():
    rate = 48000
    clean = probes.default_material(rate, 2.0)
    damage = degrade.by_name("clean")

    engine = SlowFirstCall()
    loaded = Loaded(engine)
    probes.warm_up(loaded, rate)
    assert engine.calls >= 1, "warm_up did not call the engine"

    run = probes.run_all(loaded, clean, rate, damage)
    speed = run.value("throughput", "realtime")
    assert speed is not None

    # The same engine measured without a warm-up pays the cold call.
    cold_engine = SlowFirstCall()
    cold_run = probes.run_all(Loaded(cold_engine), clean, rate, damage)
    cold_speed = cold_run.value("throughput", "realtime")

    assert speed > cold_speed * 2, (
        f"warmed {speed:.0f}x vs cold {cold_speed:.0f}x — the warm-up "
        f"did not take the first-call cost out of the measurement"
    )
