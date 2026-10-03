"""AudioSR — latent-diffusion bandwidth extension to 48 kHz.

    EARSHOT_AUDIOSR_PYTHON=/path/to/venv/bin/python earshot bench --engine audiosr
    earshot bench --engine audiosr:25          # DDIM steps; 50 is the CLI default

The widely cited reference point, and the heaviest candidate: a 6.2 GB
checkpoint and 50 diffusion steps per call. Kept for comparison, and to see
whether fewer steps cost anything audible — the only speed lever that does
not change the model.

**It runs in its own Python.** AudioSR pins numpy<=1.23.5, librosa 0.9.2 and
transformers 4.30.2, none of which install beside this project. The engine
starts ``audiosr_worker.py`` under the interpreter named in
``EARSHOT_AUDIOSR_PYTHON`` and passes audio over files and a pipe; the
contract is checked here on the way back as for any engine. ``scripts/
colab_job.py`` builds that interpreter on a Colab VM when asked.

**Pinned weights.** The library downloads its checkpoint unpinned; the worker
is handed the file fetched and verified here instead. The speech checkpoint
is used, as earshot is about speech. Its model page states no licence (the
general one is Apache-2.0); it is used for evaluation only.

**Seeded.** Diffusion starts from noise; the seed is upstream's default, 42,
so a run is repeatable and ``stability`` reports this wiring, not the model.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..fetch import Asset, ensure_all
from . import EngineError, Loaded, register

_REVISION = "413f1d734411663e95310c17d381279a0c049960"
ASSETS: tuple[Asset, ...] = (
    Asset(
        "audiosr-speech/pytorch_model.bin",
        f"https://huggingface.co/haoheliu/audiosr_speech/resolve/{_REVISION}/"
        "pytorch_model.bin",
        "59fd60a9b93346c698788bac9b6077977a21733e7a78db9d3c3af3f1bdebc640",
        "AudioSR speech checkpoint, 6.2 GB, licence unstated",
    ),
)

WORKER = Path(__file__).with_name("audiosr_worker.py")
OUTPUT_RATE = 48000
DEFAULT_STEPS = 50
SEED = 42


class AudioSREngine:
    """AudioSR in a separate interpreter, behind the engine contract."""

    def __init__(self, steps: int = DEFAULT_STEPS):
        python = os.environ.get("EARSHOT_AUDIOSR_PYTHON")
        if not python:
            raise EngineError(
                "audiosr runs in its own Python, which AudioSR's pins require: "
                "make a Python 3.10 environment with `pip install audiosr==0.0.7` "
                "and point EARSHOT_AUDIOSR_PYTHON at its interpreter "
                "(scripts/colab.py --audiosr does this on Colab)"
            )
        (checkpoint,) = ensure_all(ASSETS)
        self.steps = steps
        self.name = "audiosr" if steps == DEFAULT_STEPS else f"audiosr@{steps}"
        self._scratch = tempfile.TemporaryDirectory(prefix="earshot-audiosr-")
        self.process_handle = subprocess.Popen(
            [python, str(WORKER), "--ckpt", str(checkpoint), "--model", "speech",
             "--steps", str(steps), "--seed", str(SEED)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            # A notebook kernel exports its inline plotting backend, and
            # AudioSR's older matplotlib refuses that name at import — the
            # fourth Colab run died of it. AudioSR draws nothing anyway.
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        if self._reply() != "ready":
            self.close()
            raise EngineError("the AudioSR worker did not start; its output is above")

    def _reply(self) -> str:
        # The library prints progress on stdout too; the protocol's lines are
        # the ones that start with its words.
        for line in self.process_handle.stdout:
            line = line.strip()
            if line in ("ready", "ok") or line.startswith("error:"):
                return line
        return "worker exited"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        import soundfile as sf

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return x
        source = Path(self._scratch.name) / "in.wav"
        target = Path(self._scratch.name) / "out.wav"
        sf.write(source, x, rate, subtype="FLOAT")
        self.process_handle.stdin.write(f"{source}\t{target}\n")
        self.process_handle.stdin.flush()
        reply = self._reply()
        if reply != "ok":
            raise EngineError(f"AudioSR failed: {reply}")
        out, out_rate = sf.read(target, dtype="float32")
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if out_rate != rate:
            from math import gcd

            from scipy import signal

            factor = gcd(int(out_rate), int(rate))
            out = signal.resample_poly(out, rate // factor, out_rate // factor).astype(
                np.float32
            )
        # AudioSR pads its input to a 2.5 s grid and returns the padded
        # length; the tail is cut so the start stays aligned.
        if len(out) >= len(x):
            return np.ascontiguousarray(out[: len(x)])
        return np.concatenate([out, np.zeros(len(x) - len(out), dtype=np.float32)])

    def close(self) -> None:
        """Stop the worker and free its GPU memory."""
        if self.process_handle is not None:
            if self.process_handle.stdin:
                self.process_handle.stdin.close()
            try:
                self.process_handle.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process_handle.kill()
            self.process_handle = None
        self._scratch.cleanup()

    def __del__(self):
        if getattr(self, "process_handle", None) is not None:
            self.close()


@register("audiosr")
def _load(argument: str) -> Loaded:
    """``audiosr`` or ``audiosr:<DDIM steps>``."""
    try:
        steps = int(argument) if argument else DEFAULT_STEPS
    except ValueError:
        steps = 0
    if steps < 1:
        raise EngineError(f"audiosr takes a number of DDIM steps, got {argument!r}")
    started = time.perf_counter()
    engine = AudioSREngine(steps)
    return Loaded(
        engine,
        load_seconds=time.perf_counter() - started,
        notes={"steps": steps, "seed": SEED, "revision": _REVISION},
    )
