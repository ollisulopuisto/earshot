"""Invent where there is nothing, keep what is there.

    earshot bench --engine 'router:lavasr'

The idea, in the owner's words: measure the energy in each band, invent if
there is nothing there, and leave it alone if there is. It resolves the
awkward fork in this project — a telephone-band guest has nothing left to
filter, so only invention helps; a good local microphone has everything, so
invention means replacing a voice that was fine — by refusing to choose. The
same engine does both, and the material decides which.

**Where the decision is made.** Not per frequency bin. A bin-by-bin blend of
two complex spectra cancels wherever both have content and their phases
disagree, and a generative engine regenerates phase by definition. Instead
the router finds, for each moment, the frequency above which the input holds
nothing but noise floor — its *band edge* — and crossfades from the input to
the engine's output across a narrow region around it. Below the edge the
speaker's own waveform survives untouched; above it, invented content fills
a space that was empty anyway. This is what LavaSR does at a fixed 4 kHz;
here the edge moves to where the material actually stops.

**How the edge is found.** Per analysis window, the noise floor is the median
level of the top of the spectrum, where speech has nothing to contribute. The
edge is the highest frequency whose level still stands ``EDGE_MARGIN_DB``
above that floor. Silence has no edge and is left alone entirely — a passage
with no speech in it should not have speech invented into it.

**Why the edge is smoothed.** An edge recomputed every frame jitters between
neighbouring bins on ordinary speech, and a crossover that moves quickly is
audible as a warble. It is smoothed over ``EDGE_SMOOTH_S`` and can only move
so fast, so it tracks a codec change over a sentence and ignores a sibilant.

This engine is a hypothesis with a knob, not a finished answer. What it
should do is show up in the bench as high ``origin`` correlation in bands
that had content and low correlation only where the input was empty. If it
does not, it is wrong and the numbers will say so.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, check_contract, register

# Analysis window for the edge. Long enough to average over a syllable,
# short enough to follow a guest whose connection changes mid-episode.
FRAME = 2048
HOP = 512

# The edge is found as a cliff relative to the speech band, not as a level
# above a noise floor. Measuring the floor at the top of the spectrum was the
# first attempt and it fails on exactly the input this exists for: a codec
# empties the top, so "the floor" becomes digital silence and every stopband
# ripple stands above it. The reference is instead the level in the band that
# always has speech in it.
REFERENCE_BAND_HZ = (300.0, 3000.0)

# How far below the speech band a frequency must fall to count as empty.
# Sixty decibels is a cliff, not a roll-off: a dull microphone that is merely
# quiet up top keeps its content, and only a band that was actually zeroed
# gets invented into. The asymmetry is deliberate — replacing a voice that
# was fine is the expensive mistake.
CLIFF_DB = 60.0

# The crossfade around the edge, in octaves. Narrow enough that little is
# blended, wide enough not to be a step.
CROSSOVER_OCTAVES = 0.5

# How quickly the edge may move. A jittering crossover warbles.
EDGE_SMOOTH_S = 0.75

# An edge above this is not worth acting on: the input is already full-band
# and there is nothing to invent.
EDGE_CEILING_HZ = 15000.0

# Below this the input is too damaged for the premise to hold — if a signal
# has nothing above 1 kHz it is not band-limited speech, it is broken.
EDGE_FLOOR_HZ = 1000.0


class RouterEngine:
    """Keeps the input where it has content, uses an engine where it does not."""

    def __init__(self, inner, margin_db: float = CLIFF_DB):
        self.inner = inner
        self.margin_db = margin_db
        self.name = f"router({inner.engine.name})"
        self.edges: list[float] = []  # viimeisimmän ajon reunat, mittausta varten

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from scipy import signal as sig

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) < FRAME * 2:
            # Too short to measure an edge on; pass the engine's own answer.
            return check_contract(x, self.inner.engine.process(x, rate), self.name)

        generated = check_contract(
            x, self.inner.engine.process(x, rate), self.inner.engine.name
        )

        window = np.hanning(FRAME).astype(np.float32)
        _, _, X = sig.stft(x, rate, nperseg=FRAME, noverlap=FRAME - HOP, window=window)
        _, _, G = sig.stft(
            generated, rate, nperseg=FRAME, noverlap=FRAME - HOP, window=window
        )
        frames = min(X.shape[1], G.shape[1])
        X, G = X[:, :frames], G[:, :frames]
        freqs = np.fft.rfftfreq(FRAME, 1 / rate)

        edge, leave_alone = self._band_edge(X, freqs, rate)
        self.edges = edge.tolist()
        mask = self._crossfade(edge, freqs)  # (bins, frames), 0 = input, 1 = engine
        # A frame with nothing to fill is passed through whole. Trusting the
        # crossover alone was not enough: with the edge at its ceiling the
        # mask still opened above it, and digital silence came back as the
        # engine's noise at -18 dBFS.
        mask[:, leave_alone] = 0.0
        blended = X * (1.0 - mask) + G * mask

        _, out = sig.istft(
            blended, rate, nperseg=FRAME, noverlap=FRAME - HOP, window=window
        )
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if len(out) >= len(x):
            return np.ascontiguousarray(out[: len(x)])
        return np.concatenate([out, np.zeros(len(x) - len(out), dtype=np.float32)])

    def _band_edge(self, X: np.ndarray, freqs: np.ndarray, rate: int) -> np.ndarray:
        """Highest frequency per frame that still holds content, in Hz.

        Returns ``(edge, leave_alone)``. ``leave_alone`` marks frames that
        must be passed through untouched: silence, which must not be given a
        voice, and material whose content already reaches the top, where
        there is no empty band to fill.
        """
        power = 20.0 * np.log10(np.abs(X) + 1e-9)
        ceiling = min(EDGE_CEILING_HZ, rate / 2 * 0.95)

        band = (freqs >= REFERENCE_BAND_HZ[0]) & (freqs <= REFERENCE_BAND_HZ[1])
        if band.sum() < 4:
            return np.full(X.shape[1], ceiling), np.ones(X.shape[1], dtype=bool)
        reference = np.median(power[band], axis=0)

        # A frame whose speech band is itself at the noise floor has nothing
        # to extend. -80 dBFS is far below any recorded speech and far above
        # digital silence.
        silent = reference < -80.0

        occupied = power > (reference - CLIFF_DB)[None, :]
        index = np.where(
            occupied.any(axis=0),
            occupied.shape[0] - 1 - np.argmax(occupied[::-1], axis=0),
            len(freqs) - 1,
        )
        edge = freqs[np.clip(index, 0, len(freqs) - 1)]
        edge = np.clip(edge, EDGE_FLOOR_HZ, ceiling)
        edge[silent] = ceiling

        # Smooth in time: the crossover may drift over a sentence, not jump
        # between syllables.
        span = max(1, int(EDGE_SMOOTH_S * rate / HOP))
        kernel = np.ones(span) / span
        padded = np.pad(edge, (span, span), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="same")[span:-span]
        # Smoothing must not drag a silent frame back down into inventing.
        smoothed[silent] = ceiling
        # Nothing to fill: either no speech, or speech that already reaches
        # the top of the spectrum.
        leave_alone = silent | (smoothed >= ceiling * 0.999)
        return smoothed, leave_alone

    def _crossfade(self, edge: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """0 below the edge, 1 above it, raised-cosine in between."""
        with np.errstate(divide="ignore", invalid="ignore"):
            octaves = np.log2(
                np.maximum(freqs[:, None], 1.0) / np.maximum(edge[None, :], 1.0)
            )
        half = CROSSOVER_OCTAVES / 2.0
        ramp = np.clip((octaves + half) / max(CROSSOVER_OCTAVES, 1e-6), 0.0, 1.0)
        return (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)


@register("router")
def _load(argument: str) -> Loaded:
    """``router:<engine>`` or ``router:<engine>@<margin dB>``."""
    from . import load as load_engine

    spec, _, margin = argument.rpartition("@")
    if not spec:
        spec, margin = argument, ""
    if not spec:
        raise EngineError("router needs an engine, e.g. router:lavasr")
    try:
        threshold = float(margin) if margin else CLIFF_DB
    except ValueError:
        raise EngineError(f"router margin must be a number, got {margin!r}") from None

    started = time.perf_counter()
    inner = load_engine(spec)
    return Loaded(
        RouterEngine(inner, threshold),
        load_seconds=time.perf_counter() - started,
        notes={"inner": spec, "margin_db": threshold},
    )
