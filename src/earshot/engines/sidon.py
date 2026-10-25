"""Sidon — speech restoration by resynthesis from w2v-BERT 2.0 features.

    earshot bench --engine sidon

The research survey's first pick (docs/research/2026-10-03-speech-restoration-
survey.md): MIT code and weights, 48 kHz output, and a self-reported speaker
similarity of 0.961 against Miipher's 0.930 on LibriTTS. It works like
UniPASE in kind — damaged speech becomes self-supervised features and a
decoder speaks them again — so expect the same two risks the bench found for
UniPASE until measured: harm to clean audio, and identity drift.

Wired as upstream's own demo does it (the sidon_demo_beta Space): peak to
0.9, a 50 Hz high-pass, 16 kHz into the SeamlessM4T filterbank front end,
the TorchScript feature extractor and decoder in 96 s chunks with the last
feature frame carried across, 960 samples trimmed per chunk, and the 48 kHz
result cut to length. The demo's own peak normalisation is undone at the
end so the take keeps the input's level; the bench matches loudness anyway,
but a 0.9-peak output would look like a level change to every other probe.
"""

from __future__ import annotations

import time

import numpy as np

from ..fetch import Asset, ensure_all
from . import EngineError, Loaded, register, torch_device

_REVISION = "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
_BASE = f"https://huggingface.co/sarulab-speech/sidon-v0.1/resolve/{_REVISION}/"
_W2V_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
_DIGESTS = {
    "feature_extractor_cpu.pt": "fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019",
    "decoder_cpu.pt": "34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4",
    "feature_extractor_cuda.pt": "c3739c332da4ce00cc9019a328bcb19ee8f863ac5eb0b52386672772aab836ce",
    "decoder_cuda.pt": "a789c71f54ca7db4b28becdc87c5e8b9260cb568cca462d502c54900db74a1cc",
}
_PREPROCESSOR = Asset(
    "sidon/w2v-bert-2.0/preprocessor_config.json",
    f"https://huggingface.co/facebook/w2v-bert-2.0/resolve/{_W2V_REVISION}/"
    "preprocessor_config.json",
    "8e6281aad64f97e40534135a59dcc5d33571efae376f2a25adf5551951897ab4",
    "w2v-BERT 2.0 filterbank settings, MIT",
)


def assets_for(device: str) -> tuple[Asset, ...]:
    """The TorchScript files are exported per device; fetch only one set."""
    kind = "cuda" if device.startswith("cuda") else "cpu"
    return tuple(
        Asset(f"sidon/{name}", _BASE + name, _DIGESTS[name], f"Sidon v0.1 {name}, MIT")
        for name in (f"feature_extractor_{kind}.pt", f"decoder_{kind}.pt")
    ) + (_PREPROCESSOR,)


# For the pinning test and for fetching everything by hand.
ASSETS: tuple[Asset, ...] = assets_for("cpu") + assets_for("cuda")[:2]

MODEL_RATE = 16000
OUTPUT_RATE = 48000
CHUNK_S = 96
PAD_DELAY = 160 * OUTPUT_RATE // MODEL_RATE  # 480 samples, 10 ms


class SidonEngine:
    name = "sidon"

    def __init__(self):
        try:
            import torch
            import torchaudio  # noqa: F401
            import transformers
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the sidon engine needs the 'sidon' extra: pip install earshot[sidon]"
            ) from exc
        self.device = torch_device()
        if self.device == "mps":
            self.device = "cpu"  # exported for cpu and cuda only
        paths = ensure_all(assets_for(self.device))
        try:
            self.feature_extractor = torch.jit.load(str(paths[0]), map_location=self.device)
            self.decoder = torch.jit.load(str(paths[1]), map_location=self.device)
            self.preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
                str(paths[2].parent)
            )
        except Exception as exc:
            raise EngineError(f"could not start Sidon: {exc}") from exc

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        import torch
        import torchaudio

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return x
        peak = float(np.abs(x).max())
        if peak < 1e-9:
            return np.zeros_like(x)
        scale = 0.9 / peak
        wav = torch.from_numpy(x * scale).view(1, -1)
        wav = torchaudio.functional.highpass_biquad(wav, rate, 50)
        wav = torchaudio.functional.resample(wav, rate, MODEL_RATE)
        wav = torch.nn.functional.pad(wav, (0, 24000))
        pieces, carried = [], None
        try:
            with torch.inference_mode():
                for chunk in wav.view(-1).split(MODEL_RATE * CHUNK_S):
                    inputs = self.preprocessor(
                        torch.nn.functional.pad(chunk, (160, 160)).numpy(),
                        sampling_rate=MODEL_RATE, return_tensors="pt",
                    )
                    feature = self.feature_extractor(
                        inputs["input_features"].to(self.device))["last_hidden_state"]
                    if carried is not None:
                        feature = torch.cat([carried, feature], dim=1)
                    pieces.append(self.decoder(feature.transpose(1, 2)).view(-1)[:-960].cpu())
                    carried = feature[:, -1:]
        except Exception as exc:
            raise EngineError(f"Sidon failed: {exc}") from exc
        # The 160 samples of left padding (10 ms at 16 kHz) come out as 10 ms of
        # delay: measured on three EARS excerpts the speech envelope ran
        # 332-402 samples late with the padding and 64-137 samples early
        # without it. Upstream's demo keeps the padding and the delay; the
        # padding is kept here too, and its known length removed.
        out = torch.cat(pieces).numpy().astype(np.float32)[PAD_DELAY:] / scale
        if rate != OUTPUT_RATE:
            from math import gcd

            from scipy import signal

            factor = gcd(OUTPUT_RATE, int(rate))
            out = signal.resample_poly(out, rate // factor, OUTPUT_RATE // factor).astype(
                np.float32)
        n = len(x)
        if len(out) >= n:
            return np.ascontiguousarray(out[:n])
        return np.concatenate([out, np.zeros(n - len(out), dtype=np.float32)])


@register("sidon")
def _load(argument: str) -> Loaded:
    """``sidon``."""
    if argument:
        raise EngineError(f"sidon takes no options, got {argument!r}")
    started = time.perf_counter()
    engine = SidonEngine()
    return Loaded(engine, load_seconds=time.perf_counter() - started,
                  notes={"revision": _REVISION, "device": engine.device})
