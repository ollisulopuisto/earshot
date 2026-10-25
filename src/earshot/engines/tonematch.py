"""Tone match: pull an engine's long-term balance back to the speaker's own.

    earshot bench --engine tonematch:sidon

Built for Sidon, which the owner preferred blind but heard as changing the
speaker's tone. Measured against the originals on twelve pp53 comparisons
(2026-10-04), relative to 300–3000 Hz: body +2.0 to +2.5 dB, 800–2500 Hz
+2.0 to +3.4, everything above 2.5 kHz 1 to 3.4 dB darker. Pitch was right
on average (within 8 cents) but its frame-to-frame wobble fell to 0.65–0.81
of the original's — that part no EQ can touch.

The reference is the input, which is what exists in production: below the
input's band edge its long-term spectrum is the speaker's (plus whatever
damage, which a long-term average mostly shrugs off). There the output is
equalised, in third-octave smoothing and at most ±``LIMIT_DB``, to the
input's balance. Above the edge the input has nothing to say, so the
engine's invented top is left as it made it, with half an octave of fade.
Zero-phase, so nothing moves in time.

One measured caution: an EQ match did not recover UniPASE's speaker-
embedding similarity (+0.000, 2026-10-04). This targets what the owner
hears as tone, which may not be what the embedding measures.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, register

LIMIT_DB = 6.0
LOWEST_HZ = 80.0
FADE_OCTAVES = 0.5


def _smooth(freqs: np.ndarray, power: np.ndarray, fraction: int = 3) -> np.ndarray:
    out = np.empty_like(power)
    for i, fc in enumerate(freqs):
        lo, hi = fc * 2 ** (-1 / (2 * fraction)), fc * 2 ** (1 / (2 * fraction))
        band = (freqs >= lo) & (freqs <= hi)
        out[i] = power[band].mean() if band.any() else power[i]
    return out


def match_gain(reference: np.ndarray, output: np.ndarray, rate: int,
               top_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and a linear gain taking ``output``'s balance to
    ``reference``'s between LOWEST_HZ and ``top_hz``, fading to 1 above."""
    from scipy import signal

    n = min(8192, len(reference))
    f, pr = signal.welch(np.asarray(reference, dtype=np.float64), rate, nperseg=n)
    _, po = signal.welch(np.asarray(output, dtype=np.float64), rate, nperseg=n)
    gain_db = 10 * np.log10(np.maximum(_smooth(f, pr), 1e-30) / np.maximum(_smooth(f, po), 1e-30))
    # A level difference is not a tone difference: centre on the speech band.
    speech = (f >= 300) & (f <= 3000)
    gain_db -= np.median(gain_db[speech]) if speech.any() else 0.0
    gain_db = np.clip(gain_db, -LIMIT_DB, LIMIT_DB)
    inside = (f >= LOWEST_HZ) & (f <= top_hz)
    if not inside.any():
        return f, np.ones_like(f)
    low_edge = gain_db[inside][0]
    gain_db[f < LOWEST_HZ] = low_edge
    above = f > top_hz
    octaves = np.log2(np.maximum(f[above], 1.0) / top_hz)
    fade = np.clip(1 - octaves / FADE_OCTAVES, 0.0, 1.0)
    gain_db[above] = gain_db[inside][-1] * fade
    return f, 10 ** (gain_db / 20)


class ToneMatchEngine:
    def __init__(self, inner, inner_name: str | None = None):
        self.inner = inner
        self.name = f"tonematch({inner_name or getattr(inner, 'name', 'engine')})"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from .universr import content_edge

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        y = np.asarray(self.inner.process(x, rate), dtype=np.float32)
        if len(x) < 4096 or not np.any(x) or not np.any(y):
            return y
        top = min(content_edge(x, rate) * 0.9, 16000.0, rate / 2 * 0.95)
        f, gain = match_gain(x, y, rate, top)
        spectrum = np.fft.rfft(y.astype(np.float64))
        bins = np.fft.rfftfreq(len(y), 1 / rate)
        out = np.fft.irfft(spectrum * np.interp(bins, f, gain), len(y))
        return out.astype(np.float32)


@register("tonematch")
def _load(argument: str) -> Loaded:
    """``tonematch:<engine spec>``."""
    from . import load as load_engine

    if not argument:
        raise EngineError("tonematch needs an engine, e.g. tonematch:sidon")
    started = time.perf_counter()
    inner = load_engine(argument)
    return Loaded(ToneMatchEngine(inner.engine, inner.engine.name),
                  load_seconds=time.perf_counter() - started,
                  notes={"inner": argument, "limit_db": LIMIT_DB})
