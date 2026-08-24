"""LavaSR — bandwidth extension and denoising, Apache-2.0.

Vocos-shaped: mel spectrogram in, an eight-layer backbone, an ISTFT head out.
The inference itself is vendored and pinned (``earshot.vendor``); this file is
only the wiring that makes it an engine the bench can measure.

Three things about it are worth knowing before reading its numbers.

**It works at 16 kHz in and 48 kHz out.** Whatever it is given is resampled
down, enhanced, and resampled back. So on already-full-band material it is
not "restoring" anything above 8 kHz — it is *replacing* it with what the
model thinks belongs there. Expect the ``origin`` probe to say so.

**It keeps the original below 4 kHz.** Upstream's ``FastLRMerge`` crossfades
the model's output over the untouched input at 4 kHz, so the speaker's own
waveform survives where most of the intelligibility lives, and only the top
is generated. That is a deliberately conservative design and the main reason
this is the first candidate rather than a full resynthesiser.

**Denoising is optional and off by default.** It is a separate model and a
separate question; leaving it off measures bandwidth extension alone, which
is what the candidate is for. ``lavasr:denoise`` turns it on.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..fetch import Asset, ensure_all
from . import EngineError, Loaded, register

# Model weights live on a GitHub release rather than in the repository, and
# each is pinned by digest: a file that changed upstream would turn every
# number measured with it into a number about a different model.
ASSETS: tuple[Asset, ...] = (
    Asset(
        "lavasr/denoiser_core_legacy_fixed63.onnx",
        "https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/"
        "denoiser_core_legacy_fixed63.onnx",
        "8afa7f4db9f356f7bfb575bb207d8673a728a7baf6773e0b10226a5e15687f2a",
        "LavaSR denoiser, 1.8 MB",
    ),
    Asset(
        "lavasr/enhancer_backbone.onnx",
        "https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/"
        "enhancer_backbone.onnx",
        "841e96d261dffdf1dc974f3d29e2cfcf1b16fd0b358749c1ace0bbfa1d4c8ddd",
        "LavaSR enhancer graph",
    ),
    Asset(
        "lavasr/enhancer_backbone.onnx.data",
        "https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/"
        "enhancer_backbone.onnx.data",
        "a125a4ede7cfdd1073d906a3cadf2171a30be6a40f296ad28772e0ba258de8c5",
        "LavaSR enhancer weights, 52 MB",
    ),
    Asset(
        "lavasr/enhancer_spec_head.onnx",
        "https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/"
        "enhancer_spec_head.onnx",
        "f66fd164c55fd1b07e5cea5e687c71522b192f452691128fd7ae4e6b26dbc683",
        "LavaSR ISTFT head graph",
    ),
    Asset(
        "lavasr/enhancer_spec_head.onnx.data",
        "https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/"
        "enhancer_spec_head.onnx.data",
        "b855e309b027af9aa75285b97b345571b6bd695a30fde434d06c979d83885fd6",
        "LavaSR ISTFT head weights, 4.3 MB",
    ),
)

MODEL_RATE = 16000
OUTPUT_RATE = 48000


class LavaSREngine:
    """LavaSR behind the engine contract."""

    def __init__(self, denoise: bool = False, threads: int = 0):
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the lavasr engine needs the 'lavasr' extra: "
                "pip install earshot[lavasr]"
            ) from exc

        paths = dict(zip((a.name for a in ASSETS), ensure_all(ASSETS)))
        # The ONNX graphs reference their external weight files by the exact
        # name they were built with, so the cached copies keep upstream's
        # names and live together in one subdirectory. Renaming them gives
        # "external data path does not exist" at session load.
        from ..vendor import lavasr_core

        config = Path(lavasr_core.__file__).parent / "lavasr_config.yaml"
        self.denoise = denoise
        self.name = "lavasr+denoise" if denoise else "lavasr"
        try:
            self.model = lavasr_core.LavaSR(
                config=str(config),
                denoiser_onnx=str(paths["lavasr/denoiser_core_legacy_fixed63.onnx"]),
                enhancer_backbone_onnx=str(paths["lavasr/enhancer_backbone.onnx"]),
                enhancer_spec_head_onnx=str(paths["lavasr/enhancer_spec_head.onnx"]),
                ort_intra_op_num_threads=threads or 0,
                ort_inter_op_num_threads=1,
            )
        except Exception as exc:
            raise EngineError(f"could not start LavaSR: {exc}") from exc

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from scipy import signal

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        wanted = len(x)
        if wanted == 0:
            return x

        # A signal shorter than the model's own window comes back empty or
        # ragged. Pad up, process, and cut back — the bench slices material
        # and a chain engine passes fragments, so this is not hypothetical.
        floor = MODEL_RATE  # one second at model rate, generous
        padded = x
        if len(padded) * MODEL_RATE // max(rate, 1) < floor:
            padded = np.concatenate([x, np.zeros(rate, dtype=np.float32)])

        down = _resample(padded, rate, MODEL_RATE)
        try:
            out = self.model.enhance(down, apply_denoise=self.denoise)
        except Exception as exc:
            raise EngineError(f"LavaSR failed: {exc}") from exc
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        back = _resample(out, OUTPUT_RATE, rate)

        # Resampling twice lands within a few samples of the original length;
        # the contract wants it exact, so trim or pad the tail. Doing it at
        # the tail rather than the head keeps the start aligned, which is
        # what every probe compares.
        if len(back) >= wanted:
            return np.ascontiguousarray(back[:wanted])
        return np.concatenate([back, np.zeros(wanted - len(back), dtype=np.float32)])


def _resample(x: np.ndarray, source: int, target: int) -> np.ndarray:
    from math import gcd

    from scipy import signal

    if source == target:
        return np.asarray(x, dtype=np.float32)
    factor = gcd(int(source), int(target))
    return signal.resample_poly(x, target // factor, source // factor).astype(
        np.float32
    )


@register("lavasr")
def _load(argument: str) -> Loaded:
    """``lavasr`` or ``lavasr:denoise``."""
    options = {part for part in argument.split(",") if part}
    unknown = options - {"denoise"}
    if unknown:
        raise EngineError(f"unknown lavasr option(s): {', '.join(sorted(unknown))}")
    started = time.perf_counter()
    engine = LavaSREngine(denoise="denoise" in options)
    return Loaded(
        engine,
        load_seconds=time.perf_counter() - started,
        notes={"denoise": engine.denoise, "model_rate": MODEL_RATE},
    )
