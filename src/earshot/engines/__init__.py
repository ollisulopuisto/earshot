"""Engines: anything that takes damaged speech and hands back speech.

One interface, so the bench never knows what it is measuring. A VST3
plug-in, an ONNX model, a shell command wrapping someone's inference
script — all the same shape:

    engine = load("vst3:/Library/Audio/Plug-Ins/VST3/Thing.vst3")
    restored = engine.process(audio, 48000)

Three rules an engine must keep, because the bench compares sample against
sample and cannot check what it cannot align:

1. **The output has the same number of samples as the input.** An engine
   that trims its own latency reports as damage everything it shifted.
2. **The output is aligned to the input.** If the engine has latency, it
   compensates it. ``Engine.process`` is offline; there is no excuse.
3. **One channel in, one channel out.** Restoration is per microphone. A
   stereo file is two problems, not one.

An engine that cannot keep these declares it by raising ``EngineError``,
which the bench reports rather than swallowing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class EngineError(RuntimeError):
    """The engine could not run, or could not keep its side of the contract."""


class Engine(Protocol):
    """What the bench requires of anything it measures."""

    name: str

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray: ...


@dataclass
class Loaded:
    """An engine plus what it cost to get it.

    ``load_seconds`` is separate from processing time on purpose: a model
    that takes ten seconds to load and then runs at 200× realtime is a
    different proposition from one that loads instantly and crawls, and a
    single "how fast is it" number hides the difference.
    """

    engine: Engine
    load_seconds: float = 0.0
    notes: dict = field(default_factory=dict)


_REGISTRY: dict = {}


def register(scheme: str):
    """Register a loader for ``scheme:rest`` specifications."""

    def decorate(function):
        _REGISTRY[scheme] = function
        return function

    return decorate


def load(spec: str) -> Loaded:
    """Turn ``"scheme:argument"`` into a running engine.

    The scheme is the part before the first colon, so a path with colons in
    it still works. ``"passthrough"`` alone is valid: schemes may take no
    argument.
    """
    scheme, _, argument = spec.partition(":")
    if scheme not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise EngineError(f"unknown engine scheme {scheme!r}; known: {known}")
    return _REGISTRY[scheme](argument)


def schemes() -> list[str]:
    return sorted(_REGISTRY)


def check_contract(before: np.ndarray, after: np.ndarray, name: str) -> np.ndarray:
    """Enforce rule 1 at the point of use.

    Checked here rather than trusted, because a length change is silent:
    every later number would still compute, and every one of them would be
    measuring an offset rather than a restoration.
    """
    after = np.asarray(after)
    if after.ndim != 1:
        raise EngineError(f"{name} returned {after.ndim} channels, expected 1")
    if len(after) != len(before):
        raise EngineError(
            f"{name} changed the sample count: {len(before)} in, {len(after)} out"
        )
    if not np.all(np.isfinite(after)):
        raise EngineError(f"{name} returned non-finite samples")
    return after


from . import lavasr as _lavasr  # noqa: E402,F401
from . import passthrough as _passthrough  # noqa: E402,F401  (registers itself)
from . import vst3 as _vst3  # noqa: E402,F401
