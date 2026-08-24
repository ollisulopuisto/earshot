"""Reading and writing, with the bench's assumptions in one place.

Mono float32 throughout. Restoration is per microphone: a stereo file is
two problems, and averaging them would invent a third.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read(path: str | Path, seconds: float = 0.0, start: float = 0.0):
    """Read a file as mono float32. Returns ``(audio, rate)``."""
    info = sf.info(str(path))
    offset = int(start * info.samplerate)
    frames = int(seconds * info.samplerate) if seconds else -1
    data, rate = sf.read(str(path), start=offset, frames=frames, dtype="float32")
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), rate


def write(path: str | Path, audio: np.ndarray, rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), rate, subtype="PCM_24")
