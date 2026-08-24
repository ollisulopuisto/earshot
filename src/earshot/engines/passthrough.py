"""The control: hands the audio back untouched.

Not a joke entry. Every probe should be run against it, because a metric
that scores passthrough as an improvement is measuring the wrong thing, and
a metric that scores it as damage is broken. It is also the honest baseline
for "is this engine worth its runtime at all".

``gain`` exists for the same reason: an engine that only turns the signal up
should be detectable, so the bench needs something that only turns the
signal up.
"""

from __future__ import annotations

import numpy as np

from . import Loaded, register


class Passthrough:
    def __init__(self, gain_db: float = 0.0):
        self.gain_db = gain_db
        self.name = "passthrough" if not gain_db else f"gain{gain_db:+g}dB"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        if not self.gain_db:
            return np.asarray(audio, dtype=np.float32).copy()
        return (np.asarray(audio, dtype=np.float32) * 10 ** (self.gain_db / 20)).astype(
            np.float32
        )


@register("passthrough")
def _load(argument: str) -> Loaded:
    return Loaded(Passthrough(float(argument) if argument else 0.0))
