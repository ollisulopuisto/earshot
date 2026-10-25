"""How far UniverSR chunked is from UniverSR whole, on one excerpt.

    EARSHOT_DEVICE=cuda uv run python scripts/measure_chunking.py material/local/ears

Chunking was turned on for Colab because one 8 s call needed more than a
T4's memory. It changes the output at the seams, so before chunked takes are
listened to beside whole-file ones the difference is measured: per band, the
level of (chunked - whole) under the whole-file signal.
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from earshot import degrade, engines

source = Path(sys.argv[1]) / "p001 freeform 01.wav"
x, rate = sf.read(source, dtype="float32", start=75 * 48000, frames=8 * 48000)
damaged = degrade.by_name("narrowband-voip").apply(x, rate)

os.environ.pop("EARSHOT_UNIVERSR_CHUNK_S", None)
whole = engines.load("universr").engine.process(damaged, rate)
os.environ["EARSHOT_UNIVERSR_CHUNK_S"] = "4"
chunked = engines.load("universr").engine.process(damaged, rate)


def level(a):
    return 10 * np.log10(np.mean(np.asarray(a, dtype=np.float64) ** 2) + 1e-20)


print("universr chunked (4 s, 1 s overlap) against whole, narrowband-voip, EARS p001 75-83 s")
for low, high in [(50, 1000), (1000, 4000), (4000, 8000), (8000, 16000), (16000, 23900)]:
    sos = signal.butter(6, (low, high), btype="band", fs=rate, output="sos")
    w, c = signal.sosfiltfilt(sos, whole), signal.sosfiltfilt(sos, chunked)
    print(f"  {low:5d}-{high:5d} Hz: difference {level(c - w) - level(w):+6.1f} dB under the signal")
print(f"  whole band: {level(chunked - whole) - level(whole):+6.1f} dB")
