"""DNSMOS P.835: a non-intrusive guess at speech, background and overall
quality, without a clean reference.

Microsoft's DNS Challenge model (CC BY 4.0), as its own reference script
runs it: 16 kHz audio, 9.01 s windows hopped by 1 s (shorter input repeated
to length), the raw outputs mapped through its published polynomials.

Here as a detector for the gate, not as a score to chase: it is trained on
16 kHz speech, knows nothing above 8 kHz, and the research survey found it
blind to content hallucination. What it does see, measured on five pp53
excerpts (2026-10-04): room reverberation drops SIG to 1.22–2.25 against
2.62–3.51 for clean audio, and hiss drops BAK to 2.64–3.64 against
3.55–3.95 — the two damages the gate's first detectors could not see.
"""

from __future__ import annotations

import numpy as np

from .engines import EngineError
from .fetch import Asset, ensure_all

_COMMIT = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
ASSETS: tuple[Asset, ...] = (
    Asset(
        "dnsmos/sig_bak_ovr.onnx",
        f"https://github.com/microsoft/DNS-Challenge/raw/{_COMMIT}/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
        "DNSMOS P.835, 1.2 MB, CC BY 4.0",
    ),
)

RATE = 16000
WINDOW_S = 9.01
_POLY = (
    np.poly1d([-0.08397278, 1.22083953, 0.0052439]),    # SIG
    np.poly1d([-0.13166888, 1.60915514, -0.39604546]),  # BAK
    np.poly1d([-0.06766283, 1.11546468, 0.04602535]),   # OVRL
)
_session = None


def _model():
    global _session
    if _session is None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise EngineError("DNSMOS needs onnxruntime: pip install earshot[speaker]") from exc
        (path,) = ensure_all(ASSETS)
        _session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _session


def available() -> bool:
    try:
        _model()
    except EngineError:
        return False
    return True


def dnsmos(audio: np.ndarray, rate: int) -> dict[str, float]:
    """SIG, BAK and OVRL, each 1–5, averaged over 9.01 s windows."""
    from math import gcd

    from scipy import signal

    session = _model()
    factor = gcd(int(rate), RATE)
    y = signal.resample_poly(np.asarray(audio, dtype=np.float64).reshape(-1),
                             RATE // factor, rate // factor).astype(np.float32)
    window = int(WINDOW_S * RATE)
    if not len(y):
        raise ValueError("no audio")
    while len(y) < window:
        y = np.concatenate([y, y])
    scores = []
    for start in range(0, int(len(y) / RATE - WINDOW_S) * RATE + 1, RATE):
        raw = session.run(None, {"input_1": y[start:start + window][None, :]})[0][0]
        scores.append([p(v) for p, v in zip(_POLY, raw)])
    sig, bak, ovrl = np.mean(scores, axis=0)
    return {"sig": float(sig), "bak": float(bak), "ovrl": float(ovrl)}
