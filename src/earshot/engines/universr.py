"""UniverSR — bandwidth extension by flow matching on the spectrum itself.

    earshot bench --engine universr            # band read from the input
    earshot bench --engine universr:16         # content up to 8 kHz
    earshot bench --engine router:universr

No vocoder: the model generates complex STFT bins above the input's band and
keeps the bins below it, so the speaker's own spectrum survives where it was
present. That is the property ``origin`` should confirm, and the reason this
is queued ahead of the resynthesisers. The speech-only checkpoint is used;
the general-audio one is for music and effects, which this project is not.

**It must be told the input's band**, as one of four rates: 8, 12, 16 or
24 kHz, meaning content up to 4, 6, 8 or 12 kHz. It low-passes the input to
that band before generating, so the choice is not advisory — a band below
the content discards speech. Left unset, the band is read from the input and
rounded *down* to the nearest supported one: an empty strip below the band
would be kept as silence, which is worse than discarding a little speech
that ``router:universr`` puts back.

**Full-band input is passed through.** Content reaching ``FULL_BAND_HZ`` has
nothing missing, and the widest band the model accepts would low-pass the
top half of a good microphone away to make room for its own guess.

**The seed is fixed.** Flow matching starts from noise; upstream draws fresh
noise each call unless given a seed. A fixed one makes the bench repeatable
and the ``stability`` probe will report it as deterministic, which describes
this wiring, not the model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..fetch import Asset, ensure_all
from . import EngineError, Loaded, register
from .router import CLIFF_DB, REFERENCE_BAND_HZ

# Pinned to a commit of the Hugging Face repository, not to main: the digest
# is what proves the file, the commit is what makes a mismatch mean
# corruption rather than an upstream update.
_REVISION = "225fa7cfea2c802d047c598826226befc21376dc"
_BASE = f"https://huggingface.co/woongzip1/universr-speech/resolve/{_REVISION}/"
ASSETS: tuple[Asset, ...] = (
    Asset(
        "universr-speech/config.yaml",
        _BASE + "config.yaml",
        "2e035dacc833368319f9d5ba1a34e5efd343faf749008d433e309f3c65f4dac9",
        "UniverSR speech model configuration",
    ),
    Asset(
        "universr-speech/pytorch_model.bin",
        _BASE + "pytorch_model.bin",
        "d29130a1c1558dbacaa89c0555bb34dd353c2e2c3a76e91eff876aa13166bc9b",
        "UniverSR speech weights, 229 MB, CC BY 4.0",
    ),
)

MODEL_RATE = 48000
SUPPORTED_KHZ = (8, 12, 16, 24)

# Above this the input is treated as full-band. The router's own ceiling,
# for the same reason: clean podcast microphones measure −46 dB at 16 kHz
# relative to the speech band, inside the router's 40 dB cliff, so their
# edge reads at or above it.
FULL_BAND_HZ = 15000.0

SEED = 0
# Upstream's defaults: four midpoint steps, guidance 1.5.
ODE_STEPS = 4
GUIDANCE = 1.5

# Measured on an M1 Max CPU, 2 s of audio: 15 s per model pass, and the
# defaults make sixteen (four midpoint steps, each twice for guidance), so an
# 8 s excerpt takes about 12 minutes. The Apple GPU was 11 s per pass, not
# worth the extra failure mode. Most of the time is convolutions: this torch
# build has no oneDNN on macOS, so they all take the slow path.

_FRAME = 2048
_HOP = 1024


def content_edge(audio: np.ndarray, rate: int) -> float:
    """The highest frequency the speech in ``audio`` reaches, in Hz.

    The router's rule applied to the whole excerpt at once: the long-term
    spectrum of the frames that hold speech, and the highest frequency still
    within ``CLIFF_DB`` of the 300–3000 Hz band. Silence has no edge and
    reports the top of the spectrum, which means "leave alone".
    """
    from scipy import signal

    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    top = rate / 2.0
    if len(x) < _FRAME:
        return top
    freqs, _, Z = signal.stft(x, fs=rate, nperseg=_FRAME, noverlap=_FRAME - _HOP)
    power = np.abs(Z) ** 2
    band = (freqs >= REFERENCE_BAND_HZ[0]) & (freqs <= REFERENCE_BAND_HZ[1])
    reference = 10.0 * np.log10(power[band].mean(axis=0) + 1e-20)
    # The router's silence threshold: far below recorded speech, far above
    # digital zero, which platform audio is 8 to 58 per cent of.
    voiced = reference > -80.0
    if not voiced.any():
        return top
    spectrum = 10.0 * np.log10(power[:, voiced].mean(axis=1) + 1e-20)
    level = 10.0 * np.log10(power[band][:, voiced].mean() + 1e-20)
    occupied = np.nonzero(spectrum > level - CLIFF_DB)[0]
    return float(freqs[occupied[-1]]) if len(occupied) else top


def input_rate_for(edge_hz: float) -> int:
    """The widest supported band that does not reach above ``edge_hz``."""
    fitting = [khz for khz in SUPPORTED_KHZ if khz * 500.0 <= edge_hz]
    return (fitting[-1] if fitting else SUPPORTED_KHZ[0]) * 1000


class UniverSREngine:
    """UniverSR's speech checkpoint behind the engine contract."""

    def __init__(
        self,
        fixed_rate: int | None = None,
        device: str = "cpu",
        ode_steps: int = ODE_STEPS,
        guidance: float | None = GUIDANCE,
        ode_method: str = "midpoint",
    ):
        try:
            import torch  # noqa: F401
            from universr import UniverSR
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the universr engine needs the 'universr' extra: "
                "pip install earshot[universr]"
            ) from exc
        paths = ensure_all(ASSETS)
        folder = Path(paths[0]).parent
        try:
            self.model = UniverSR.from_pretrained(str(folder), device=device)
        except Exception as exc:
            raise EngineError(f"could not start UniverSR: {exc}") from exc
        self.fixed_rate = fixed_rate
        self.ode_steps, self.guidance, self.ode_method = ode_steps, guidance, ode_method
        self.name = f"universr@{fixed_rate // 1000}k" if fixed_rate else "universr"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return x
        if self.fixed_rate:
            band = self.fixed_rate
        else:
            edge = content_edge(x, rate)
            if edge >= FULL_BAND_HZ:
                return x.copy()
            band = input_rate_for(edge)

        import torch

        source = x if rate == MODEL_RATE else _resample(x, rate, MODEL_RATE)
        try:
            out = self.model.enhance(
                torch.from_numpy(np.ascontiguousarray(source)),
                input_sr=band,
                ode_method=self.ode_method,
                ode_steps=self.ode_steps,
                guidance_scale=self.guidance,
                seed=SEED,
            )
        except Exception as exc:
            raise EngineError(f"UniverSR failed: {exc}") from exc
        out = np.asarray(out.reshape(-1).cpu().numpy(), dtype=np.float32)
        if rate != MODEL_RATE:
            out = _resample(out, MODEL_RATE, rate)
        # The inverse STFT can land a frame short; the tail is where to pad,
        # so the start stays aligned.
        if len(out) >= len(x):
            return np.ascontiguousarray(out[: len(x)])
        return np.concatenate([out, np.zeros(len(x) - len(out), dtype=np.float32)])


def _resample(x: np.ndarray, source: int, target: int) -> np.ndarray:
    from math import gcd

    from scipy import signal

    factor = gcd(int(source), int(target))
    return signal.resample_poly(x, target // factor, source // factor).astype(
        np.float32
    )


@register("universr")
def _load(argument: str) -> Loaded:
    """``universr`` or ``universr:<8|12|16|24>``, the input's rate in kHz."""
    fixed = None
    if argument:
        try:
            khz = int(argument)
        except ValueError:
            khz = None
        if khz not in SUPPORTED_KHZ:
            raise EngineError(
                f"universr takes the input's band as 8, 12, 16 or 24 (kHz), "
                f"got {argument!r}"
            )
        fixed = khz * 1000
    started = time.perf_counter()
    engine = UniverSREngine(fixed)
    return Loaded(
        engine,
        load_seconds=time.perf_counter() - started,
        notes={"input_rate": fixed or "auto", "seed": SEED, "ode_steps": ODE_STEPS,
               "revision": _REVISION},
    )
