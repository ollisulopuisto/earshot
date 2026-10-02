"""UniPASE — denoise, bandwidth extension and packet loss concealment in one.

    earshot bench --engine unipase
    earshot bench --engine unipase:noplc

The only candidate that claims packet loss. A resynthesiser end to end: the
input becomes WavLM features (a 24-layer self-supervised speech model,
fine-tuned to see through damage), an adapter cleans them, a Vocos vocoder
speaks them at 16 kHz, and a TF-GridNet post-net extends that to 48 kHz. The
post-net keeps the *vocoder's* output below 8 kHz, not the input's, so no
sample of the speaker's own waveform survives. Expect ``origin`` near zero
everywhere; whether the voice still sounds like its owner is the question
only listening and speaker similarity can answer.

**Vendored, not installed.** Upstream is a research repository, not a
package, so its inference code is copied into ``earshot/vendor/unipase`` and
pinned file by file. Its post-net imports espnet for an abstract base class
and a name-to-activation lookup; ``_shim_espnet`` supplies both rather than
installing a speech toolkit for them. If espnet is installed, it is used.

**No peak normalisation.** Upstream's script rescales the output to the
input's peak. The bench matches loudness itself, and a peak match would make
one transient decide the level of the whole file.

**Packet loss concealment is on by default**, as upstream has it.
``unipase:noplc`` turns it off: real podcast material measured 0.0 to 0.3
per cent packet loss, so whether it helps there or only invents is worth
hearing both ways.
"""

from __future__ import annotations

import sys
import time
import types

import numpy as np

from ..fetch import Asset, ensure_all
from . import EngineError, Loaded, register

_REVISION = "f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0"
_BASE = f"https://huggingface.co/Xiaobin-Rong/unipase/resolve/{_REVISION}/"
ASSETS: tuple[Asset, ...] = (
    Asset(
        "unipase/DeWavLM-Omni.pt",
        _BASE + "DeWavLM-Omni.pt",
        "16c94d05ff6ef2bdf08b0be2853826afda687c2d63c833382198db7682476689",
        "UniPASE encoder (WavLM-Large, fine-tuned), 1.26 GB",
    ),
    Asset(
        "unipase/Adapter.pt",
        _BASE + "Adapter.pt",
        "3f3f9f8076e304bc740fef5496f91528ca9ea6be7f5a63deec71cdc33471c8c7",
        "UniPASE adapter, 458 MB",
    ),
    Asset(
        "unipase/Vocoder_DWO-L1.pt",
        _BASE + "Vocoder_DWO-L1.pt",
        "7595d48962c296bab2f7cb307f299acf5a02ec6a972d9ab7da3ed71b4a54a3d5",
        "UniPASE vocoder, 455 MB",
    ),
    Asset(
        "unipase/PostNet.pt",
        _BASE + "PostNet.pt",
        "1bf950dd2432e3cd2ef7055226b259bb3ac3fba8d7ce1f5f2424581a3d19cf63",
        "UniPASE bandwidth-extension post-net, 11 MB",
    ),
)

RATE = 48000


def _shim_espnet() -> None:
    """Supply the two espnet names the vendored post-net imports."""
    try:
        import espnet2.enh.separator.abs_separator  # noqa: F401
        import espnet2.torch_utils.get_layer_from_string  # noqa: F401

        return
    except ImportError:
        pass

    import torch

    class AbsSeparator(torch.nn.Module):
        """espnet's is an ABC over nn.Module; inference needs only the base."""

    def get_layer(name: str):
        # espnet looks the name up case-insensitively among torch.nn's layers.
        layers = {key.lower(): getattr(torch.nn, key) for key in dir(torch.nn)}
        try:
            return layers[name.lower()]
        except KeyError:
            raise ValueError(f"no torch.nn layer called {name!r}") from None

    modules = {
        "espnet2": types.ModuleType("espnet2"),
        "espnet2.enh": types.ModuleType("espnet2.enh"),
        "espnet2.enh.separator": types.ModuleType("espnet2.enh.separator"),
        "espnet2.enh.separator.abs_separator": types.ModuleType(
            "espnet2.enh.separator.abs_separator"
        ),
        "espnet2.torch_utils": types.ModuleType("espnet2.torch_utils"),
        "espnet2.torch_utils.get_layer_from_string": types.ModuleType(
            "espnet2.torch_utils.get_layer_from_string"
        ),
    }
    modules["espnet2.enh.separator.abs_separator"].AbsSeparator = AbsSeparator
    modules["espnet2.torch_utils.get_layer_from_string"].get_layer = get_layer
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


class UniPASEEngine:
    """UniPASE behind the engine contract."""

    def __init__(self, plc: bool = True):
        try:
            import omegaconf  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise EngineError(
                "the unipase engine needs the 'unipase' extra: "
                "pip install earshot[unipase]"
            ) from exc
        _shim_espnet()
        paths = dict(zip((a.name for a in ASSETS), ensure_all(ASSETS)))
        try:
            from ..vendor.unipase.unipase import UniPASE

            self.model = UniPASE(
                dewavlm_ckpt_path=str(paths["unipase/DeWavLM-Omni.pt"]),
                adapter_ckpt_path=str(paths["unipase/Adapter.pt"]),
                vocoder_ckpt_path=str(paths["unipase/Vocoder_DWO-L1.pt"]),
                postnet_ckpt_path=str(paths["unipase/PostNet.pt"]),
            ).eval()
        except Exception as exc:
            raise EngineError(f"could not start UniPASE: {exc}") from exc
        self.plc = plc
        self.name = "unipase" if plc else "unipase-noplc"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        import torch

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return x
        # WavLM's convolutional front end refuses anything shorter than a few
        # of its frames: 2048 samples raised "kernel size can't be greater
        # than actual input size". Pad short input to a second of silence at
        # the tail, so the start stays aligned, and cut it back after.
        source = x
        if len(x) < rate:
            source = np.concatenate([x, np.zeros(rate - len(x), dtype=np.float32)])
        try:
            with torch.inference_mode():
                out = self.model(
                    torch.from_numpy(np.ascontiguousarray(source)).unsqueeze(0),
                    sr_in=rate,
                    sr_out=rate,
                    enable_plc=self.plc,
                )
        except Exception as exc:
            raise EngineError(f"UniPASE failed: {exc}") from exc
        # Upstream pads or trims to the input length itself; this only
        # guards the contract if that ever changes.
        out = np.asarray(out.reshape(-1).cpu().numpy(), dtype=np.float32)
        if len(out) >= len(x):
            return np.ascontiguousarray(out[: len(x)])
        return np.concatenate([out, np.zeros(len(x) - len(out), dtype=np.float32)])


@register("unipase")
def _load(argument: str) -> Loaded:
    """``unipase`` or ``unipase:noplc``."""
    if argument not in ("", "noplc"):
        raise EngineError(f"unipase takes 'noplc' or nothing, got {argument!r}")
    started = time.perf_counter()
    engine = UniPASEEngine(plc=argument != "noplc")
    return Loaded(
        engine,
        load_seconds=time.perf_counter() - started,
        notes={"plc": engine.plc, "revision": _REVISION},
    )
