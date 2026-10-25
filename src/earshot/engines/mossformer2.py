"""MossFormer2 SE 48K — a discriminative 48 kHz speech enhancer.

    earshot bench --engine mossformer2

ClearerVoice-Studio's (modelscope, Apache-2.0 code and weights): 180 Kaldi
filterbank features (60 mels with deltas and delta-deltas) in, a mask over
the input's own 1920-point STFT out. Because it masks rather than speaks,
what survives is the speaker's own waveform — the research survey's case
for a discriminative front end, since URGENT's discriminative entries kept
speaker similarity best while generative ones drifted.

The network is vendored (``earshot.vendor.mossformer2_se``); the decoding is
upstream's ``decode_one_audio_mossformer2_se_48k`` one-pass path, rewritten
here without librosa and joblib. Two deliberate differences:

* **Seeded.** Upstream computes the filterbank with ``dither=1.0`` and no
  seed, so every call differs. The seed is fixed per call, keeping
  upstream's dither (the model was trained with it) and the bench's
  repeatability.
* **Long input in chunks.** Upstream decodes up to 20 s in one pass and
  slides a 4 s window beyond that; here anything over 20 s goes through
  ``process_in_chunks`` with 20 s chunks and 1 s of crossfade.
"""

from __future__ import annotations

import time

import numpy as np

from ..fetch import Asset, ensure
from . import EngineError, Loaded, process_in_chunks, register, torch_device

_REVISION = "eff8c97925c8bec812af707814b3e5d777fd4503"
ASSETS: tuple[Asset, ...] = (
    Asset(
        "mossformer2-se-48k/last_best_checkpoint.pt",
        f"https://huggingface.co/alibabasglab/MossFormer2_SE_48K/resolve/{_REVISION}/"
        "last_best_checkpoint.pt",
        "03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6",
        "MossFormer2 SE 48K weights, 222 MB, Apache-2.0",
    ),
)

RATE = 48000
WIN, HOP, FFT, MELS = 1920, 384, 1920, 60
MAX_WAV = 32768.0
ONE_PASS_S = 20.0
SEED = 0


class MossFormer2Engine:
    name = "mossformer2"

    def __init__(self):
        try:
            import torch
            import torchaudio  # noqa: F401
            from ..vendor.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the mossformer2 engine needs the 'mossformer2' extra: "
                f"pip install earshot[mossformer2] ({exc})"
            ) from exc
        self.device = torch_device()
        if self.device == "mps":
            self.device = "cpu"
        path = ensure(ASSETS[0])
        try:
            self.model = MossFormer2_SE_48K(None).model
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            weights = checkpoint.get("model", checkpoint)
            state = self.model.state_dict()
            loaded = 0
            for key in state:
                for candidate in (key, key.replace("module.", ""), "module." + key):
                    if candidate in weights and weights[candidate].shape == state[key].shape:
                        state[key] = weights[candidate]
                        loaded += 1
                        break
            if loaded < len(state):
                raise EngineError(f"only {loaded} of {len(state)} weights matched the network")
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"could not start MossFormer2: {exc}") from exc

    def _one_pass(self, x: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio

        audio = torch.from_numpy(x.astype(np.float32) * MAX_WAV)
        torch.manual_seed(SEED)
        fbanks = torchaudio.compliance.kaldi.fbank(
            audio.unsqueeze(0), dither=1.0, frame_length=WIN / RATE * 1000,
            frame_shift=HOP / RATE * 1000, num_mel_bins=MELS, sample_frequency=RATE,
            window_type="hamming")
        tr = fbanks.transpose(0, 1)
        delta = torchaudio.functional.compute_deltas(tr)
        delta2 = torchaudio.functional.compute_deltas(delta)
        features = torch.cat([fbanks, delta.transpose(0, 1), delta2.transpose(0, 1)], dim=1)
        with torch.no_grad():
            mask = self.model(features.unsqueeze(0).to(self.device))[-1].cpu()
        window = torch.hamming_window(WIN, periodic=False)
        spectrum = torch.stft(audio, FFT, HOP, WIN, center=False, window=window,
                              return_complex=True)
        masked = spectrum * mask.permute(2, 1, 0)[..., 0]
        out = torch.istft(masked, FFT, HOP, WIN, window=window, center=False, length=len(audio))
        return (out.numpy() / MAX_WAV).astype(np.float32)

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) < WIN * 2:
            return x.copy()
        if rate != RATE:
            raise EngineError(f"mossformer2 works at 48 kHz only, got {rate} Hz")
        if len(x) > ONE_PASS_S * RATE:
            from types import SimpleNamespace

            whole = SimpleNamespace(name=self.name, process=lambda a, r: self._one_pass(a))
            return process_in_chunks(whole, x, rate, chunk_seconds=ONE_PASS_S, overlap_seconds=1.0)
        try:
            out = self._one_pass(x)
        except Exception as exc:
            raise EngineError(f"MossFormer2 failed: {exc}") from exc
        n = len(x)
        return out[:n] if len(out) >= n else np.concatenate([out, np.zeros(n - len(out), np.float32)])


@register("mossformer2")
def _load(argument: str) -> Loaded:
    """``mossformer2``."""
    if argument:
        raise EngineError(f"mossformer2 takes no options, got {argument!r}")
    started = time.perf_counter()
    engine = MossFormer2Engine()
    return Loaded(engine, load_seconds=time.perf_counter() - started,
                  notes={"revision": _REVISION, "device": engine.device})
