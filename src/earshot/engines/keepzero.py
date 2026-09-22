"""Leave the platform's digital silence as silence.

    earshot bench --engine 'keepzero:lavasr'

Remote-platform audio is 8 to 58 per cent exact digital zero: the platform
gates the pauses before the file ever arrives. A generative engine given
that input invents into the gaps — measured, LavaSR adds 34.54 dB to the
noise floor on `platform-upload`, because the hard edges into silence look
to it like something to extend.

This wrapper runs any engine and then puts the silence back: wherever the
*input* held a run of exact zeros long enough to be a gate rather than a
zero crossing, the output is zero too, with short fades so the restored
speech does not end in a click. Everything else is the inner engine's
output, untouched.

It is a guess about what an editor wants, and it may be the wrong one: a
gate's silence is itself a defect, and an engine that filled it with room
tone would arguably be restoring something. This keeps the defect rather
than letting a model decide what the room sounded like. The listening test
is where that gets settled, not here.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, check_contract, register

# A run of zeros this long is a gate, not a zero crossing. Eight samples is
# the measure the material survey used: the studio tracks hold no such run
# in 224 minutes, every platform file holds hundreds of thousands.
MIN_ZEROS = 8

# "Zero" is anything at or below -120 dBFS, not only exact zero. A gate
# followed by any filter — the platform's own resampler, or this bench's
# band limit — leaves the edges of each gap ringing down through values like
# 1e-9 rather than jumping to 0.0. Measured on one gated EARS passage (8 s,
# 28.5 per cent exact zero, 29.7 per cent at or below 1e-6), energy in the
# gaps: LavaSR alone -74.8 dBFS; wrapped, counting exact zeros only, -87.1;
# counting everything at or below 1e-6, -96.1. Eight samples in a row this
# quiet do not happen in a recording that was not silenced digitally.
#
# For comparison, router(lavasr) on the same passage leaves -117.2 dBFS in
# the gaps, because it passes silent frames through whole. Where the router
# is used, this wrapper adds nothing; it exists for an engine used alone.
SILENT = 1e-6

# The fade at each edge of a restored silence.
FADE_S = 0.005


class KeepZeroEngine:
    def __init__(self, inner):
        self.inner = inner
        self.name = f"keepzero({inner.engine.name})"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        y = np.array(
            check_contract(x, self.inner.engine.process(x, rate), self.inner.engine.name),
            dtype=np.float32, copy=True,
        )
        zero = np.abs(x) <= SILENT
        if not zero.any():
            return y
        edges = np.diff(np.concatenate(([0], zero.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        keep = np.ones(len(x), dtype=np.float32)
        fade = max(1, int(FADE_S * rate))
        ramp = (0.5 + 0.5 * np.cos(np.pi * np.arange(fade) / fade)).astype(np.float32)
        for start, stop in zip(starts, stops):
            if stop - start < MIN_ZEROS:
                continue
            keep[start:stop] = 0.0
            # Fade out over the end of the speech before the gap and in over
            # the start of the speech after it: the gap itself stays exact.
            lo = max(0, start - fade)
            keep[lo:start] = np.minimum(keep[lo:start], ramp[fade - (start - lo):])
            hi = min(len(x), stop + fade)
            keep[stop:hi] = np.minimum(keep[stop:hi], ramp[::-1][: hi - stop])
        return y * keep


@register("keepzero")
def _load(argument: str) -> Loaded:
    from . import load as load_engine

    if not argument:
        raise EngineError("keepzero wraps an engine, e.g. keepzero:lavasr")
    started = time.perf_counter()
    inner = load_engine(argument)
    return Loaded(KeepZeroEngine(inner), load_seconds=time.perf_counter() - started,
                  notes={"inner": argument})
