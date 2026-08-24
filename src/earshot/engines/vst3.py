"""Any VST3 or Audio Unit, through pedalboard.

This is how a commercial plug-in gets into the bench next to open models.
It is also how the reference numbers in ``docs/dxrevive.md`` were produced.

Two things learned the hard way and encoded here:

``reset=True`` is not optional. With ``reset=False`` pedalboard returns a
result shorter than the input by the plug-in's own latency — measured at
4641 samples with one restoration plug-in — which is a silent violation of
the engine contract and reads as damage in every metric.

The plug-in is loaded once per engine instance and never shared across
threads. A VST3 object holds state; two threads in one instance interleave
their audio.
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import EngineError, Loaded, register


class VST3Engine:
    def __init__(self, path: str, parameters: dict | None = None):
        # The path is checked before the import so that a typo reports as a
        # typo. "install the extra" is unhelpful advice when the real problem
        # is that the plug-in is somewhere else.
        if not os.path.exists(path):
            raise EngineError(f"no plug-in at {path}")
        try:
            import pedalboard
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the vst3 engine needs the 'vst3' extra: pip install earshot[vst3]"
            ) from exc
        try:
            self.plugin = pedalboard.load_plugin(path)
        except Exception as exc:
            raise EngineError(f"could not load {os.path.basename(path)}: {exc}") from exc
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.applied = self._apply(parameters or {})

    def _apply(self, parameters: dict) -> dict:
        """Set named parameters, and report what actually took.

        pedalboard's plug-in object accepts any attribute, so an unknown
        name would look like it worked and change nothing. The names are
        checked against the plug-in's own list first.
        """
        known = getattr(self.plugin, "parameters", None) or {}
        applied = {}
        for name, value in parameters.items():
            if name not in known:
                raise EngineError(
                    f"{self.name} has no parameter {name!r}; "
                    f"it has: {', '.join(sorted(known))}"
                )
            setattr(self.plugin, name, value)
            applied[name] = getattr(self.plugin, name)
        return applied

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        block = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        return self.plugin.process(block, rate, reset=True)[0]


@register("vst3")
def _load(argument: str) -> Loaded:
    """``vst3:/path/to/Thing.vst3`` or ``vst3:/path/to/Thing.vst3?mix=100``."""
    path, _, query = argument.partition("?")
    parameters = {}
    for pair in filter(None, query.split("&")):
        key, _, value = pair.partition("=")
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value
    started = time.perf_counter()
    engine = VST3Engine(path, parameters)
    return Loaded(
        engine,
        load_seconds=time.perf_counter() - started,
        notes={"parameters": engine.applied},
    )
