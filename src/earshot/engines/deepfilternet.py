"""DeepFilterNet — denoising, and only denoising.

    earshot bench --engine deepfilternet

The missing half of every result so far. LavaSR extends a band and cannot
denoise; dxRevive denoises well and deletes everything that is not speech.
This is a dedicated denoiser, so the interesting question is not whether it
beats them at their own jobs but whether it cleans *without* the collateral
damage — which the `preservation` probe will answer in one number.

**Why the reference implementation and not the ONNX export.** An ONNX export
exists, but DeepFilterNet's inference is stateful: ERB banding, a recurrent
core, and a multi-frame deep filter applied to complex bins. Porting that by
hand is a day's careful work in which every mistake shows up as a quality
difference rather than a crash — and a bench that mismeasures a candidate is
worse than one that lacks it. So this uses upstream's own code, at the cost
of a PyTorch dependency behind an extra.

**Native at 48 kHz**, which is the rate everything here works at, so no
resampling is involved and the length is exact without correction.

**The torchaudio shim** below is not optional. DeepFilterNet 0.5 imports
``torchaudio.backend.common``, which torchaudio removed. Pinning an old
torchaudio would drag an old torch with it; a five-line stand-in for a
dataclass the enhancement path never uses costs nothing and keeps the
dependency current. If upstream fixes the import, delete it.
"""

from __future__ import annotations

import dataclasses
import sys
import time
import types

import numpy as np

from . import EngineError, Loaded, register

MODEL_RATE = 48000


def _shim_torchaudio() -> None:
    """Put back the module DeepFilterNet imports and torchaudio removed."""
    if "torchaudio.backend.common" in sys.modules:
        return

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend = types.ModuleType("torchaudio.backend")
    backend.common = common
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules["torchaudio.backend.common"] = common


class DeepFilterNetEngine:
    """DeepFilterNet3 behind the engine contract."""

    def __init__(self, attenuation_db: float = 0.0):
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the deepfilternet engine needs the 'deepfilternet' extra: "
                "pip install earshot[deepfilternet]"
            ) from exc
        _shim_torchaudio()
        try:
            from df.enhance import init_df
        except Exception as exc:
            raise EngineError(f"could not import DeepFilterNet: {exc}") from exc

        try:
            self.model, self.state, _ = init_df()
        except Exception as exc:
            raise EngineError(f"could not start DeepFilterNet: {exc}") from exc
        self.attenuation_db = attenuation_db
        self.name = (
            "deepfilternet"
            if not attenuation_db
            else f"deepfilternet@{attenuation_db:g}dB"
        )
        if self.state.sr() != MODEL_RATE:  # pragma: no cover - upstream constant
            raise EngineError(
                f"DeepFilterNet reports {self.state.sr()} Hz, expected {MODEL_RATE}"
            )

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        import torch
        from df.enhance import enhance

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return x

        source = x
        if rate != MODEL_RATE:
            source = _resample(x, rate, MODEL_RATE)

        try:
            out = enhance(
                self.model,
                self.state,
                torch.from_numpy(np.ascontiguousarray(source)).unsqueeze(0),
                atten_lim_db=self.attenuation_db or None,
            )
        except Exception as exc:
            raise EngineError(f"DeepFilterNet failed: {exc}") from exc
        out = np.asarray(out.squeeze(0).cpu().numpy(), dtype=np.float32)

        if rate != MODEL_RATE:
            out = _resample(out, MODEL_RATE, rate)
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


@register("deepfilternet")
def _load(argument: str) -> Loaded:
    """``deepfilternet`` or ``deepfilternet:<max attenuation in dB>``.

    The limit is worth having: a denoiser allowed unlimited attenuation
    silences the room entirely, which is the failure this project cares
    about most. Upstream calls it ``atten_lim_db``.
    """
    try:
        attenuation = float(argument) if argument else 0.0
    except ValueError:
        raise EngineError(
            f"deepfilternet takes an attenuation limit in dB, got {argument!r}"
        ) from None
    started = time.perf_counter()
    engine = DeepFilterNetEngine(attenuation)
    return Loaded(
        engine,
        load_seconds=time.perf_counter() - started,
        notes={"attenuation_db": attenuation, "rate": MODEL_RATE},
    )
