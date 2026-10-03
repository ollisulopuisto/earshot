"""Gate: send audio to a generative engine only when it is damaged.

    earshot bench --engine gate:sidon
    earshot bench --engine 'gate:chain:declip+unipase'

Both lead candidates harm clean podcast audio — on the pp53 bench (2026-10-03)
UniPASE cost −3.55 dB and Sidon −5.41 dB of log-spectral distance on input
that needed nothing. A good microphone should come out as it went in, so the
gate measures the input first and passes it through, bit for bit, unless a
detector fires.

The detectors are the ones measured to separate damage from clean pp53
audio (three excerpts of Nyman and Wancke, 2026-10-03):

* **missing band** — content edge under ``BAND_EDGE_HZ``. Clean measured
  11.0–14.5 kHz, the call damages 3.6–9.3 kHz. The margin is small: a dull
  microphone near 10 kHz would be sent for restoration.
* **clipping** — at least ``CLIPPED_PERCENT`` of samples within 0.1 % of the
  peak. Overload measured 0.9–10.8 %, everything else 0.0 %.
* **gated silence** — at least ``ZERO_PERCENT`` of samples exactly zero.
  Platform audio measured 5.4–8.1 %, clean and the calls 0.0–0.1 %.

**Blind spots, measured:** hiss and room reverberation moved none of these.
And clipping after band-limiting can hide the missing band: on a strongly
harmonic test voice, `narrowband-voip`'s clipping refilled the top to
13.4 kHz, though real pp53 speech under it read 4.3–6.4 kHz.
Hiss even raises the band edge to 24 kHz by filling the top with noise. A
learned quality estimate (DNSMOS background, UTMOS) is the next detector.

The decision is for the whole input. Applied to an episode, run it through
``process_in_chunks`` so a guest's segment and the host's are judged apart.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, register

BAND_EDGE_HZ = 10000.0
CLIPPED_PERCENT = 0.3
ZERO_PERCENT = 1.0


def reasons(audio: np.ndarray, rate: int) -> list[str]:
    """Why ``audio`` needs restoring; an empty list means leave it alone."""
    from .universr import content_edge

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    found = []
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak <= 0.0:
        return found
    edge = content_edge(x, rate)
    if edge < BAND_EDGE_HZ:
        found.append(f"band ends at {edge / 1000:.1f} kHz")
    clipped = 100.0 * float(np.mean(np.abs(x) >= 0.999 * peak))
    if clipped >= CLIPPED_PERCENT:
        found.append(f"{clipped:.1f} % of samples at the peak")
    zeros = 100.0 * float(np.mean(x == 0.0))
    if zeros >= ZERO_PERCENT:
        found.append(f"{zeros:.1f} % exact zero")
    return found


class GateEngine:
    def __init__(self, inner, inner_name: str | None = None):
        self.inner = inner
        self.name = f"gate({inner_name or getattr(inner, 'name', 'engine')})"
        self.last_reasons: list[str] = []

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.last_reasons = reasons(x, rate)
        if not self.last_reasons:
            return x.copy()
        return np.asarray(self.inner.process(x, rate), dtype=np.float32)


@register("gate")
def _load(argument: str) -> Loaded:
    """``gate:<engine spec>``."""
    from . import load as load_engine

    if not argument:
        raise EngineError("gate needs an engine, e.g. gate:sidon")
    started = time.perf_counter()
    inner = load_engine(argument)
    return Loaded(GateEngine(inner.engine, inner.engine.name),
                  load_seconds=time.perf_counter() - started,
                  notes={"inner": argument, "band_edge_hz": BAND_EDGE_HZ,
                         "clipped_percent": CLIPPED_PERCENT, "zero_percent": ZERO_PERCENT})
