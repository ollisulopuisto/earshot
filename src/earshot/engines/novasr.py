"""NovaSR — a 53 KB upsampler from 16 kHz to 48 kHz.

    earshot bench --engine novasr

Here for completeness, as the cheapest possible baseline. Its author now
points users at LavaSR instead, which is already measured. Two things to know
before reading its numbers:

**Everything goes through 16 kHz.** The input is resampled down and the model
synthesises a 48 kHz waveform from that, so content above 8 kHz is replaced
whatever was there, and nothing below it is kept from the input either: the
output is the decoder's waveform end to end. Expect ``origin`` to say so, and
``router:novasr`` to be the fairer form.

**Full precision, on the CPU.** Upstream defaults to half precision, which
is a GPU optimisation; on a CPU it is slower, not faster, and measuring the
half-precision model would measure rounding as well.
"""

from __future__ import annotations

import time

import numpy as np

from ..fetch import Asset, ensure_all
from . import EngineError, Loaded, register

_REVISION = "27c2a0f8836d80d3e10a3729f690b578a0c49e17"
ASSETS: tuple[Asset, ...] = (
    Asset(
        "novasr/pytorch_model_v1.bin",
        f"https://huggingface.co/YatharthS/NovaSR/resolve/{_REVISION}/"
        "pytorch_model_v1.bin",
        "7ca4d3a650a8e3c80fe0b2e022975c8f57f41b9f1358fc856bcce67fd18d3124",
        "NovaSR weights, 53 KB, Apache-2.0",
    ),
)

MODEL_RATE = 16000
OUTPUT_RATE = 48000


class NovaSREngine:
    """NovaSR behind the engine contract."""

    name = "novasr"

    def __init__(self):
        try:
            import torch  # noqa: F401
            from NovaSR import FastSR
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the novasr engine needs the 'novasr' extra: "
                "pip install earshot[novasr]"
            ) from exc
        (path,) = ensure_all(ASSETS)
        try:
            self.model = FastSR(ckpt_path=str(path), half=False)
        except Exception as exc:
            raise EngineError(f"could not start NovaSR: {exc}") from exc

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        import torch

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return x
        down = _resample(x, rate, MODEL_RATE)
        try:
            out = self.model.infer(
                torch.from_numpy(np.ascontiguousarray(down)).view(1, 1, -1)
            )
        except Exception as exc:
            raise EngineError(f"NovaSR failed: {exc}") from exc
        out = np.asarray(out.reshape(-1).cpu().numpy(), dtype=np.float32)
        if rate != OUTPUT_RATE:
            out = _resample(out, OUTPUT_RATE, rate)
        # Resampling twice lands within a few samples; fix the tail so the
        # start stays aligned.
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


@register("novasr")
def _load(argument: str) -> Loaded:
    """``novasr``."""
    if argument:
        raise EngineError(f"novasr takes no options, got {argument!r}")
    started = time.perf_counter()
    engine = NovaSREngine()
    return Loaded(
        engine,
        load_seconds=time.perf_counter() - started,
        notes={"revision": _REVISION},
    )
