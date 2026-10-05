"""AudioSR, run in its own Python and spoken to over a pipe.

Not imported by earshot: it is started by ``earshot.engines.audiosr`` with
the interpreter named in ``EARSHOT_AUDIOSR_PYTHON``, which has audiosr 0.0.7
installed and none of earshot. AudioSR pins numpy<=1.23.5, librosa 0.9.2 and
transformers 4.30.2, which cannot live beside this project.

Protocol: prints ``ready`` once the model is loaded, then for each line
``<input.wav>\\t<output.wav>`` on stdin writes the output and prints ``ok``,
or ``error: <message>``.

**The checkpoint is ours, not the library's.** ``download_checkpoint`` would
fetch whatever Hugging Face serves today; it is replaced here, at the call
site, by the file earshot fetched and verified against a pinned digest.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="speech")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Run as a script, this file's folder leads sys.path, and that folder
    # holds earshot's own engine module called audiosr.py — which the first
    # Colab run imported instead of the library.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]

    import audiosr.pipeline
    import soundfile as sf
    import numpy as np

    audiosr.pipeline.download_checkpoint = lambda name: args.ckpt
    model = audiosr.pipeline.build_model(model_name=args.model)
    print("ready", flush=True)

    for line in sys.stdin:
        try:
            source, target = line.rstrip("\n").split("\t")
            out = audiosr.pipeline.super_resolution(
                model, source, seed=args.seed, ddim_steps=args.steps,
                guidance_scale=args.guidance,
            )
            sf.write(target, np.asarray(out, dtype=np.float32).reshape(-1), 48000,
                     subtype="FLOAT")
            print("ok", flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the engine
            print(f"error: {type(exc).__name__}: {exc}".replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
