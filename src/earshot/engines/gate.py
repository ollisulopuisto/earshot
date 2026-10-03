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

* **speech quality and background** — DNSMOS SIG under ``SIG_BELOW`` or BAK
  under ``BAK_BELOW``, when onnxruntime is installed (``earshot.quality``).
  Added for the two damages the first three could not see.

**Blind spots of the first three, measured:** hiss and room reverberation
moved none of them.
And clipping after band-limiting can hide the missing band: on a strongly
harmonic test voice, `narrowband-voip`'s clipping refilled the top to
13.4 kHz, though real pp53 speech under it read 4.3–6.4 kHz.
Hiss even raises the band edge to 24 kHz by filling the top with noise. A
learned quality estimate (DNSMOS background, UTMOS) is the next detector.

**Measured, gate v2** (2026-10-04), share of excerpts sent for restoration:

| damage | pp53, 5 (thresholds set here) | EARS, 10 (held out) |
|---|---|---|
| clean | 0/5 | 0/10 |
| room | 5/5 | 10/10 |
| hiss | 4/5 | 4/10 |
| wideband-voip, narrowband-voip, landline, overload, overload-call | 5/5 each | 10/10 each |
| voip-call | 5/5 | 8/10 |
| platform-upload | 4/5 | 2/10 (the recipe's gate rarely closes on EARS) |

Clean audio was never touched. Hiss at 20 dB SNR is the weak spot held out.

The decision is for the whole input. Applied to an episode, run it through
``process_in_chunks`` so a guest's segment and the host's are judged apart.
"""

from __future__ import annotations

import time

import numpy as np

from . import EngineError, Loaded, register

KINDS = ("band", "zero", "clip", "sig", "bak")  # also the route's priority

BAND_EDGE_HZ = 10000.0
CLIPPED_PERCENT = 0.3
ZERO_PERCENT = 1.0
# DNSMOS, when installed. Just under the clean minimums measured on five pp53
# excerpts (SIG 2.62, BAK 3.55): room read SIG 1.22-2.25, hiss BAK 2.64-3.64.
# Five excerpts is few; these move when more material is measured.
SIG_BELOW = 2.45
BAK_BELOW = 3.50
# Hiss without a model: in the quietest tenth of frames, 1-12 kHz, hiss is
# spectrally flat. Measured 2026-10-04: hiss at 20 dB SNR 0.555-0.567;
# clean EARS at most 0.519, clean pp53 at most 0.462; rooms and calls lower
# still. Thin margin on EARS, and only white-ish hiss reads this flat.
HISS_FLATNESS = 0.54


def quiet_flatness(x: np.ndarray, rate: int) -> float:
    """Median spectral flatness, 1–12 kHz, of the quietest tenth of frames
    (exact zeros excluded); NaN when there are too few frames."""
    from scipy import signal

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 2048 * 11:  # fewer than ten frames: no verdict
        return float("nan")
    f, _, z = signal.stft(x, rate, nperseg=2048, noverlap=1024)
    power = np.abs(z) ** 2
    band = (f >= 1000) & (f <= min(12000, rate / 2 * 0.95))
    energy = power[band].sum(axis=0)
    keep = energy > 0
    if keep.sum() < 10:
        return float("nan")
    quietest = np.argsort(energy[keep])[: max(5, int(0.1 * keep.sum()))]
    q = power[band][:, keep][:, quietest] + 1e-20
    return float(np.median(np.exp(np.mean(np.log(q), axis=0)) / np.mean(q, axis=0)))


def findings(audio: np.ndarray, rate: int) -> list[tuple[str, str]]:
    """``(kind, why)`` for each detector that fired; kinds are ``KINDS``."""
    from .universr import content_edge

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    found = []
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak <= 0.0:
        return found
    edge = content_edge(x, rate)
    if edge < BAND_EDGE_HZ:
        found.append(("band", f"band ends at {edge / 1000:.1f} kHz"))
    clipped = 100.0 * float(np.mean(np.abs(x) >= 0.999 * peak))
    if clipped >= CLIPPED_PERCENT:
        found.append(("clip", f"{clipped:.1f} % of samples at the peak"))
    zeros = 100.0 * float(np.mean(x == 0.0))
    if zeros >= ZERO_PERCENT:
        found.append(("zero", f"{zeros:.1f} % exact zero"))
    flatness = quiet_flatness(x, rate)
    if flatness == flatness and flatness >= HISS_FLATNESS:
        found.append(("bak", f"hiss: quiet frames {flatness:.2f} flat"))
    from .. import quality

    if quality.available():
        scores = quality.dnsmos(x, rate)
        if scores["sig"] < SIG_BELOW:
            found.append(("sig", f"speech quality {scores['sig']:.2f} (DNSMOS SIG)"))
        if scores["bak"] < BAK_BELOW:
            found.append(("bak", f"background {scores['bak']:.2f} (DNSMOS BAK)"))
    return found


def reasons(audio: np.ndarray, rate: int) -> list[str]:
    """Why ``audio`` needs restoring; an empty list means leave it alone."""
    return [why for _, why in findings(audio, rate)]


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


# Which engine each damage goes to, where it was measured. Missing band:
# Sidon led on every call-like damage (pp53 bench, 2026-10-03). Room (SIG):
# WPE, the one engine that moved speaker similarity *towards* the speaker
# (+0.011/+0.024 on realistic rooms, 2026-10-04) while UniPASE and Sidon
# cleaned further at -0.09 to -0.11 — the owner's rule puts the voice first;
# "chain:wpe+unipase" is the stronger alternative. Gated silence keeps its
# zeros. Clipping: declip alone, +2.01 dB and speaker +0.017 on five EARS
# excerpts, where every generative engine made overload worse (UniPASE
# -0.48, Sidon -0.84 dB). Hiss: unbounded DeepFilterNet, +6.80 dB, and the
# owner's blind pick on hiss; deepfilternet:12 costs less identity (-0.014
# against -0.052) for +5.88 dB (2026-10-04).
# Then MossFormer2 (2026-10-04, five EARS excerpts): on hiss +9.07 dB,
# speaker -0.047, OVRL +0.30 against DeepFilterNet's +8.95 / -0.052 / +0.27 —
# small margins, against the owner's blind pick, so for the owner's ears to settle.
# After WPE on room-laptop: +1.28 dB, speaker +0.008, OVRL +0.81 against WPE
# alone +0.88 / +0.011 / +0.57. In front of Sidon on calls it added nothing.
DEFAULT_ROUTE = {
    "band": "sidon",
    "zero": "keepzero:unipase",
    "clip": "declip",
    "sig": "chain:wpe+mossformer2",
    "bak": "mossformer2",
}


class RouteEngine:
    """The gate, with a choice: the first finding in ``KINDS`` order picks
    the engine; no finding passes the input through untouched."""

    def __init__(self, rules: dict[str, str]):
        self.rules = dict(rules)
        self.name = "route(" + ",".join(f"{k}={v}" for k, v in self.rules.items()) + ")"
        self._loaded: dict[str, object] = {}
        self.last_choice: str | None = None
        self.last_reasons: list[str] = []

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from . import load as load_engine

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        found = findings(x, rate)
        self.last_reasons = [why for _, why in found]
        kinds = {kind for kind, _ in found}
        self.last_choice = next((self.rules[k] for k in KINDS if k in kinds and k in self.rules),
                                None)
        if self.last_choice is None:
            return x.copy()
        if self.last_choice not in self._loaded:
            self._loaded[self.last_choice] = load_engine(self.last_choice).engine
        return np.asarray(self._loaded[self.last_choice].process(x, rate), dtype=np.float32)


@register("route")
def _load_route(argument: str) -> Loaded:
    """``route`` (the measured defaults) or ``route:band=sidon,sig=unipase``."""
    rules = dict(DEFAULT_ROUTE)
    if argument:
        rules = {}
        for part in argument.split(","):
            kind, _, spec = part.partition("=")
            if kind not in KINDS or not spec:
                raise EngineError(
                    f"route takes kind=engine pairs with kinds {', '.join(KINDS)}; got {part!r}")
            rules[kind] = spec
    started = time.perf_counter()
    return Loaded(RouteEngine(rules), load_seconds=time.perf_counter() - started,
                  notes={"rules": rules})
