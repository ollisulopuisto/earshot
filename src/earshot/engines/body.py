"""Body: the lowest harmonics a telephone band took away, put back.

    earshot bench --engine body               # below 300 Hz
    earshot bench --engine 'chain:unipase+body'

The owner heard restorations without body, and the listening sets measured
it: telephone-band damage takes 80–250 Hz about 20 dB down against the speech
band, and LavaSR, NovaSR and the router pass that loss straight through,
since they keep the input below 4 kHz. Nothing on the bench extends downward.

The method is Pulakka et al.'s (Interspeech 2011; IEEE TASLP 2012), which
does exactly this for telephone speech: track the pitch, which survives in
the harmonics above the cut, and synthesise the harmonics below it as
sinusoids. Where they estimate the level with a Gaussian mixture trained on
wideband speech, this uses one measured balance: in voiced frames of clean
podcast speech, how far the band below the cut sits under the speech band
(``TARGET_DB``, and where it came from beside it). It adds only what is
missing, so a voice that kept its low end is passed through untouched.

Deterministic, no model, no dependency beyond scipy.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, register

CUTOFF_HZ = 300.0
F0_RANGE = (60.0, 400.0)
YIN_THRESHOLD = 0.2
ANALYSIS_RATE = 8000
FRAME = 320  # 40 ms at the analysis rate
HOP = 80  # 10 ms

# Leave the input alone when its low band is within this much of the
# target already: a good microphone has its body, and adding to it is boom.
ALREADY_THERE_DB = 6.0

# How far 80–300 Hz sits under 300–3000 Hz in voiced frames of clean
# speech. Measured 2026-10-03 with this module's own pitch_track and
# frame_energies:
#
#   pp53 studio tracks (nyman, wancke), 6 min, 1911 voiced frames: -2.0 dB
#   EARS (p001, p002, p008), 40 s, 1321 voiced frames:              +1.4 dB
#
# Flat on purpose. Split by pitch, the medians moved between -5 and +6 dB
# with no trend shared across speakers, and the interquartile range was
# about ±7 dB within each bucket: a pitch-dependent table would have fitted
# noise. The podcast tracks are the target material, so their median is the
# target; EARS's close anechoic microphone plausibly adds proximity bass.
TARGET_DB: tuple[tuple[float, float], ...] = ((60.0, -2.0), (400.0, -2.0))


def target_db(f0: np.ndarray | float) -> np.ndarray:
    """The low/speech balance clean speech has at pitch ``f0``."""
    pitch, level = zip(*TARGET_DB)
    return np.interp(f0, pitch, level)


def pitch_track(x: np.ndarray, rate: int) -> np.ndarray:
    """Fundamental frequency per 10 ms hop, 0 where unvoiced (YIN).

    Run at 8 kHz on purpose: a telephone band keeps the period in the
    harmonics above 300 Hz even with the fundamental gone, which is the
    whole reason the missing ones can be rebuilt.
    """
    from math import gcd

    from scipy import signal

    factor = gcd(int(rate), ANALYSIS_RATE)
    y = signal.resample_poly(np.asarray(x, dtype=np.float64),
                             ANALYSIS_RATE // factor, rate // factor)
    lags = np.arange(int(ANALYSIS_RATE / F0_RANGE[1]), int(ANALYSIS_RATE / F0_RANGE[0]) + 1)
    count = max(0, (len(y) - FRAME - lags[-1]) // HOP + 1)
    f0 = np.zeros(count)
    loudest = np.sqrt(np.mean(y**2)) + 1e-12
    for i in range(count):
        start = i * HOP
        frame = y[start: start + FRAME]
        if np.sqrt(np.mean(frame**2)) < loudest * 10 ** (-40 / 20):
            continue
        d = np.array([np.sum((frame - y[start + tau: start + tau + FRAME]) ** 2)
                      for tau in range(lags[-1] + 1)])
        cumulative = np.cumsum(d[1:]) / np.arange(1, len(d))
        normalised = np.ones_like(d)
        normalised[1:] = d[1:] / np.maximum(cumulative, 1e-12)
        window = normalised[lags]
        below = np.flatnonzero(window < YIN_THRESHOLD)
        if not len(below):
            continue
        j = below[0]
        while j + 1 < len(window) and window[j + 1] < window[j]:
            j += 1
        tau = float(lags[j])
        if 0 < j < len(window) - 1:
            a, b, c = window[j - 1], window[j], window[j + 1]
            tau += 0.5 * (a - c) / max(a - 2 * b + c, 1e-12)
        f0[i] = ANALYSIS_RATE / tau
    # A median over five hops removes single-frame octave slips.
    if count >= 5:
        voiced = f0 > 0
        smoothed = signal.medfilt(f0, 5)
        f0 = np.where(voiced & (smoothed > 0), smoothed, f0 * voiced)
    return f0


def frame_energies(x: np.ndarray, rate: int, low: float, high: float, hops: int) -> np.ndarray:
    """Mean power of ``low``–``high`` Hz in each analysis hop's frame."""
    from scipy import signal

    sos = signal.butter(4, (low, high), btype="band", fs=rate, output="sos")
    band = signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64)) ** 2
    hop = HOP * rate // ANALYSIS_RATE
    frame = FRAME * rate // ANALYSIS_RATE
    return np.array([band[i * hop: i * hop + frame].mean() if i * hop < len(band) else 0.0
                     for i in range(hops)])


class BodyEngine:
    """Synthesised low harmonics, added only where they are missing."""

    def __init__(self, cutoff: float = CUTOFF_HZ):
        self.cutoff = cutoff
        self.name = "body" if cutoff == CUTOFF_HZ else f"body@{cutoff:g}"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        f0 = pitch_track(x, rate) if len(x) else np.zeros(0)
        voiced = f0 > 0
        if voiced.sum() < 5:
            return x.copy()
        low = frame_energies(x, rate, 80.0, self.cutoff, len(f0))
        mid = frame_energies(x, rate, 300.0, 3000.0, len(f0))
        have = 10 * np.log10((low[voiced] + 1e-20) / (mid[voiced] + 1e-20))
        want = target_db(f0[voiced])
        if np.median(have - want) > -ALREADY_THERE_DB:
            return x.copy()

        need = np.zeros(len(f0))
        need[voiced] = np.maximum(mid[voiced] * 10 ** (want / 10) - low[voiced], 0.0)

        hop = HOP * rate / ANALYSIS_RATE
        centres = np.arange(len(f0)) * hop + FRAME * rate / ANALYSIS_RATE / 2
        t = np.arange(len(x))
        pitch = np.interp(t, centres[voiced], f0[voiced])
        phase = 2 * np.pi * np.cumsum(pitch) / rate
        out = x.astype(np.float64)
        for k in range(1, int(self.cutoff / F0_RANGE[0]) + 1):
            present = voiced & (k * f0 < self.cutoff) & (k * f0 >= 50.0)
            if not present.any():
                continue
            # Equal share of the missing power to each harmonic under the
            # cut in that frame; a sine of amplitude a carries a**2 / 2.
            shares = np.array([np.sum((np.arange(1, 40) * p < self.cutoff)
                                      & (np.arange(1, 40) * p >= 50.0)) if p > 0 else 1
                               for p in f0])
            amplitude = np.where(present, np.sqrt(2 * need / np.maximum(shares, 1)), 0.0)
            out += np.interp(t, centres, amplitude) * np.sin(k * phase)
        return out.astype(np.float32)


@register("body")
def _load(argument: str) -> Loaded:
    """``body`` or ``body:<cutoff Hz>``, where the band to rebuild ends."""
    try:
        cutoff = float(argument) if argument else CUTOFF_HZ
    except ValueError:
        raise EngineError(f"body takes a cutoff in Hz, got {argument!r}") from None
    started = time.perf_counter()
    return Loaded(BodyEngine(cutoff), load_seconds=time.perf_counter() - started,
                  notes={"cutoff_hz": cutoff})
