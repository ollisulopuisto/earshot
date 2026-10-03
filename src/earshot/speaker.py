"""Speaker embeddings: is this still the same person?

WeSpeaker's ResNet34 trained on VoxCeleb (CC BY 4.0), as its own ONNX
export, fed what its own inference feeds it: 16 kHz audio scaled to 16-bit
range, 80 Kaldi filterbanks of 25 ms every 10 ms with a Hamming window and no
dither, mean-normalised over time. A 256-dimensional embedding comes out;
two recordings of one person sit close in it, two people far apart.

Asked because the owner's blind favourite, UniPASE, re-speaks the whole
signal and keeps none of the input's samples. Whether the voice it speaks is
still its owner's is the question no other probe answers.
"""

from __future__ import annotations

import numpy as np

from .engines import EngineError
from .fetch import Asset, ensure_all

_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
ASSETS: tuple[Asset, ...] = (
    Asset(
        "wespeaker/voxceleb_resnet34_LM.onnx",
        f"https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/{_REVISION}/"
        "voxceleb_resnet34_LM.onnx",
        "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068",
        "WeSpeaker ResNet34 speaker embeddings, 26 MB, CC BY 4.0",
    ),
)

MODEL_RATE = 16000
_session = None


def _model():
    global _session
    if _session is None:
        try:
            import onnxruntime
            import torchaudio  # noqa: F401
        except ImportError as exc:
            raise EngineError(
                "speaker similarity needs the 'speaker' extra: pip install earshot[speaker]"
            ) from exc
        (path,) = ensure_all(ASSETS)
        _session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _session


def embed(x: np.ndarray, rate: int) -> np.ndarray:
    """A unit-length speaker embedding of ``x``."""
    from math import gcd

    import torch
    from scipy import signal
    from torchaudio.compliance import kaldi

    session = _model()
    factor = gcd(int(rate), MODEL_RATE)
    y = signal.resample_poly(np.asarray(x, dtype=np.float64), MODEL_RATE // factor,
                             rate // factor).astype(np.float32)
    feats = kaldi.fbank(torch.from_numpy(y).unsqueeze(0) * (1 << 15), num_mel_bins=80,
                        frame_length=25, frame_shift=10, dither=0.0,
                        sample_frequency=MODEL_RATE, window_type="hamming",
                        use_energy=False)
    feats = (feats - feats.mean(dim=0)).unsqueeze(0).numpy().astype(np.float32)
    name = session.get_inputs()[0].name
    vector = np.asarray(session.run(None, {name: feats})[0]).reshape(-1)
    return vector / (np.linalg.norm(vector) + 1e-12)


def available() -> bool:
    try:
        _model()
    except EngineError:
        return False
    return True
