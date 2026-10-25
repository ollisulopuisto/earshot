"""Clipped peaks, redrawn from their neighbours.

    earshot bench --engine declip

Clipping in this project's real material is rare — single-digit parts per
million — which is why this is small, local and deterministic rather than a
model. A clipped run is a plateau at the rail: several consecutive samples
at the same extreme, with the waveform that should have been there cut off.
Each plateau is replaced by a cubic through the unclipped samples either
side of it, constrained never to fall back inside the rail, because the one
thing known for certain about a clipped sample is that the true value was at
least that large.

Everything that is not on a plateau is returned exactly. A file without
clipping comes back bit for bit.

Plateaus longer than ``MAX_RUN_S`` are left alone: past a few milliseconds
the neighbours say too little about what was cut off, and a guess there is
invention. That is where a learned declipper would earn its place, and the
bench would have to show that it does.
"""

from __future__ import annotations

import numpy as np

from . import Loaded, register

# A plateau: at least this many consecutive samples at the rail.
MIN_RUN = 3

# Within this fraction of the file's peak counts as "at the rail". Real
# converters clip a hair under full scale, not exactly on it.
RAIL_FRACTION = 0.999

# Samples of context either side used for the fit.
CONTEXT = 12

# Longer plateaus are not repaired. 3 ms is about a quarter-period of a
# 80 Hz fundamental.
MAX_RUN_S = 0.003


def _runs(mask: np.ndarray):
    """(start, stop) of each run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


class DeclipEngine:
    name = "declip"

    def __init__(self):
        self.repaired = 0

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from scipy.interpolate import CubicSpline

        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        y = x.copy()
        self.repaired = 0
        peak = np.max(np.abs(x)) if len(x) else 0.0
        if peak <= 0:
            return y.astype(np.float32)
        longest = max(MIN_RUN, int(MAX_RUN_S * rate))
        rail = peak * RAIL_FRACTION

        for sign in (1.0, -1.0):
            at_rail = sign * x >= rail
            for start, stop in _runs(at_rail):
                length = stop - start
                if length < MIN_RUN or length > longest:
                    continue
                left = np.arange(max(0, start - CONTEXT), start)
                right = np.arange(stop, min(len(x), stop + CONTEXT))
                # The context must itself be unclipped, or the fit is fitted
                # to another plateau.
                left = left[sign * x[left] < rail]
                right = right[sign * x[right] < rail]
                if len(left) < 3 or len(right) < 3:
                    continue
                known = np.concatenate([left, right])
                fit = CubicSpline(known, x[known])
                inside = np.arange(start, stop)
                guess = fit(inside)
                # Never back inside the rail: the true value was at least
                # as large as what was recorded.
                y[inside] = sign * np.maximum(sign * guess, sign * x[inside])
                self.repaired += length
        return y.astype(np.float32)


@register("declip")
def _load(argument: str) -> Loaded:
    return Loaded(DeclipEngine())
