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


def process_in_chunks(
    engine,
    audio: np.ndarray,
    rate: int,
    chunk_seconds: float = 60.0,
    overlap_seconds: float = 2.0,
) -> np.ndarray:
    """Run an engine over a long recording without holding it all at once.

    Memory scales with the length of a single ``process`` call: measured with
    LavaSR in a fresh process, 0.30 GB for 5 s of audio and 1.41 GB for 160 s,
    which puts a 56 minute episode somewhere between 10 and 25 GB. That is
    more than the machine this was written on has free, so restoring a real
    recording — the thing the project exists to do — was being killed while
    the bench, which uses 10 second excerpts, ran perfectly.

    Chunks overlap and are crossfaded linearly rather than butted together.
    Linear is right here because both chunks are the same source and the
    engine should produce nearly the same thing in the overlap, so the two
    ramps sum to unity; an equal-power fade would lift the seam instead.
    The overlap has to be long enough for whatever context the engine keeps
    — the router smooths its band edge over 0.75 s — which is why the default
    is seconds and not milliseconds.

    Anything shorter than one chunk is passed straight through, so short
    signals take exactly the code path they always did.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    chunk = int(chunk_seconds * rate)
    overlap = int(overlap_seconds * rate)
    if chunk <= 0 or len(x) <= chunk:
        return check_contract(x, engine.process(x, rate), getattr(engine, "name", "?"))
    overlap = max(1, min(overlap, chunk // 2))

    # Complementary ramps, so the two halves of a crossfade sum to exactly
    # one and no normalising pass is needed. The obvious linspace(0, 1, n)
    # against its own reverse sums to (n-1)/n instead, which is close enough
    # to pass a -60 dB test and wrong enough to need a full-length weight
    # array to hide it — and that array cost 2.6 GB on a real episode.
    fade_in = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
    fade_out = (1.0 - fade_in).astype(np.float32)

    step = chunk - overlap
    out = np.zeros(len(x), dtype=np.float32)

    start = 0
    while start < len(x):
        stop = min(start + chunk, len(x))
        piece = x[start:stop]
        # Copied, not viewed. An engine that returns its input unchanged
        # hands back a view of the caller's array, and fading that in place
        # silently rewrites the recording being processed — which is how this
        # first went wrong, with 0.38 of error at every seam.
        done = np.array(
            check_contract(piece, engine.process(piece, rate),
                           getattr(engine, "name", "?")),
            dtype=np.float32, copy=True,
        )

        # Fade in everywhere except the start of the recording and out
        # everywhere except the end, so every sample is covered by weights
        # that sum to one without anything having to divide afterwards.
        if start > 0:
            done[:overlap] *= fade_in[: len(done)]
        if stop < len(x):
            tail = min(overlap, len(done))
            done[len(done) - tail:] *= fade_out[:tail]

        out[start:stop] += done
        if stop >= len(x):
            break
        start += step

    return out


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


from . import chain as _chain  # noqa: E402,F401
from . import deepfilternet as _deepfilternet  # noqa: E402,F401
from . import lavasr as _lavasr  # noqa: E402,F401
from . import passthrough as _passthrough  # noqa: E402,F401  (registers itself)
from . import router as _router  # noqa: E402,F401
from . import vst3 as _vst3  # noqa: E402,F401
