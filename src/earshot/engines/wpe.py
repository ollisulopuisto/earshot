"""WPE — classical dereverberation by weighted prediction error.

    earshot bench --engine wpe
    earshot bench --engine wpe:taps=10,delay=3,iterations=3   # nara_wpe's example

The research survey's first step for rooms (docs/research/): nara_wpe (MIT),
single channel. It predicts the late reverberation of each STFT bin from
that bin's past, ``delay`` frames back and ``taps`` deep, and subtracts it.
Linear and causal in its prediction, so it cannot remove the direct path —
the property that matters after DeepFilterNet took speech out with the echo
and the owner's blind pick on a reverberant room was the damaged take.

**Settings, measured** (2026-10-04, five EARS excerpts, rooms with a real
direct-to-reverberant ratio): nara_wpe's example settings (10 taps, delay 3,
3 iterations) barely moved anything (+0.15 dB on room-laptop). 60 taps
(320 ms of history), delay 2, 5 iterations: DNSMOS SIG 2.36 → 2.92 on
room-laptop and 3.11 → 3.36 on room-near (clean 3.46), log-spectral gain
+1.10 and +2.17 dB, and speaker similarity *up* +0.011 and +0.024. UniPASE
and Sidon clean a room further (SIG 3.5–3.6) but cost 0.09–0.11 of speaker
similarity; WPE → UniPASE halves that cost. On clean input WPE costs
−0.82 dB, which is what the gate is for.

Run at 16 kHz-sized frames scaled to the input rate: a 1024-point STFT at
48 kHz (21 ms) with a 256 hop, the frame shape nara_wpe's examples use at
16 kHz scaled up. Short input is returned untouched: with fewer frames than
taps there is no past to predict from.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, register

FFT = 1024
HOP = 256


class WPEEngine:
    def __init__(self, taps: int = 60, delay: int = 2, iterations: int = 5):
        try:
            import nara_wpe  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError("the wpe engine needs the 'wpe' extra: pip install earshot[wpe]") \
                from exc
        self.taps, self.delay, self.iterations = taps, delay, iterations
        self.name = "wpe" if (taps, delay, iterations) == (60, 2, 5) else \
            f"wpe@{taps},{delay},{iterations}"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from nara_wpe.utils import istft, stft
        from nara_wpe.wpe import wpe

        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        if len(x) < FFT * (self.taps + self.delay + 2):
            return x.astype(np.float32)
        Y = stft(x[None, :], size=FFT, shift=HOP)          # (1, T, F)
        Z = wpe(Y.transpose(2, 0, 1), taps=self.taps, delay=self.delay,
                iterations=self.iterations, statistics_mode="full").transpose(1, 2, 0)
        z = istft(Z, size=FFT, shift=HOP)[0]
        n = len(x)
        z = z[:n] if len(z) >= n else np.concatenate([z, np.zeros(n - len(z))])
        return z.astype(np.float32)


@register("wpe")
def _load(argument: str) -> Loaded:
    """``wpe`` or ``wpe:taps=60,delay=2,iterations=5``."""
    options = {"taps": 60, "delay": 2, "iterations": 5}
    for part in filter(None, argument.split(",")):
        key, _, value = part.partition("=")
        if key not in options:
            raise EngineError(f"wpe takes taps, delay, iterations; got {part!r}")
        try:
            options[key] = int(value)
        except ValueError:
            raise EngineError(f"wpe {key} must be an integer, got {value!r}") from None
    started = time.perf_counter()
    engine = WPEEngine(**options)
    return Loaded(engine, load_seconds=time.perf_counter() - started, notes=options)
