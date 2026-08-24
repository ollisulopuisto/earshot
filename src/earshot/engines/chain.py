"""Engines composed in series.

    earshot bench --engine 'chain:vst3:/path/Thing.vst3?mix=100+lavasr'

Restoration is probably not one model. The candidates are lopsided in
opposite directions — one denoises well and deletes non-speech, another
preserves everything and cannot denoise — and the interesting question is
whether running them in order beats either alone.

Composition is safe by construction here, and that is the whole reason the
contract exists: every engine returns as many samples as it was given, at
the same positions, so stage two receives exactly what stage one produced
with no alignment to negotiate. The contract is re-checked *between* stages
rather than only at the end, so a violation is attributed to the stage that
committed it instead of to the chain.

Order matters and is not commutative. Denoise before bandwidth extension so
the generator is given clean input to extend; the other way round, it
generates detail from noise and the denoiser then has to remove content that
was invented on purpose. The bench will say which is true for a given pair —
that is what it is for — but that is the reasoning behind the default advice.

Separator is ``+``. An engine spec containing a literal ``+`` (a plug-in path
with a plus in it) cannot be chained by name; give it its own scheme or
rename the file.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, check_contract, register


class ChainEngine:
    """Several engines, in order, each handed the previous one's output."""

    def __init__(self, stages: list, specs: list[str]):
        if len(stages) < 2:
            raise EngineError(
                "a chain needs at least two engines; "
                "use the engine directly if there is only one"
            )
        self.stages = stages
        self.specs = specs
        self.name = "chain(" + " → ".join(s.engine.name for s in stages) + ")"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        out = np.asarray(audio)
        for stage in self.stages:
            name = stage.engine.name
            try:
                produced = stage.engine.process(out, rate)
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"{name} failed inside the chain: {exc}") from exc
            # Checked here, not only at the end: a chain that changes length
            # should name the stage that did it, or debugging means bisecting
            # by hand.
            out = check_contract(out, produced, name)
        return out


@register("chain")
def _load(argument: str) -> Loaded:
    from . import load as load_engine

    parts = [part.strip() for part in argument.split("+") if part.strip()]
    if len(parts) < 2:
        raise EngineError(
            "a chain is two or more engine specs joined by '+', "
            "e.g. chain:passthrough+lavasr"
        )
    started = time.perf_counter()
    stages = [load_engine(part) for part in parts]
    return Loaded(
        ChainEngine(stages, parts),
        load_seconds=time.perf_counter() - started,
        notes={"stages": parts},
    )
