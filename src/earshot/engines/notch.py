"""Remove a mains buzz, and nothing else.

The second use case this project serves is a good local microphone with
something wrong that is not the voice: a live room, or a constant buzz from a
ground loop. Nothing here addressed it. LavaSR crossfades its output in above
4 kHz and cannot reach a mains family at 50 to 300 Hz however hard it tries,
and the only denoiser wired up needs an extra that is not always installed.

A ground loop does not need a neural network. It is a handful of known
frequencies, and a notch that knows the mains frequency removes them
outright. That makes this both the right tool for the job and the control
every learned denoiser should have to beat: if a model cannot do better than
this on a ground loop, the ground loop did not need the model.

What it deliberately does not do is anything else. It will not touch
reverberation, broadband noise or a hammer blow, and it should not be
reported as if it had.
"""

from __future__ import annotations

import numpy as np

from . import EngineError, Loaded, register

# Wide enough to catch the drift — mains wanders by a few hundredths of a
# hertz with grid load, and a notch narrower than the wander leaves a residue
# that warbles. Narrow enough to leave a low male fundamental alone: 85 Hz is
# well outside a Q of 30 at 50 Hz.
QUALITY = 30.0

# Six harmonics reaches 300 Hz at 50 Hz mains, which is where a ground loop
# stops being audible against speech. Going further starts costing voice.
HARMONICS = 6

# How far a harmonic must stand above its own neighbourhood before it is
# treated as hum rather than as voice. Measured in the owner's recordings, a
# real but inaudible ground loop stands 3 to 12 dB proud; an audible one is
# far higher. Ten keeps the notch off material that does not need it, which
# matters because the upper harmonics sit inside a male voice's first
# harmonic region.
DETECT_DB = 10.0


class NotchEngine:
    """Notches the mains fundamental and its harmonics, leaves the rest."""

    def __init__(self, mains_hz: float = 50.0, harmonics: int = HARMONICS):
        self.mains_hz = mains_hz
        self.harmonics = harmonics
        self.name = f"notch({mains_hz:g}Hz)"
        self._cached = None

    def _present(self, x: np.ndarray, rate: int, freq: float) -> bool:
        """Is there actually a tone here, or only voice?

        Notching unconditionally is how this first went wrong. Six harmonics
        at 50 Hz reaches 300 Hz, which is inside the first harmonic region of
        a male voice, and it took 3.7 dB out of the speech at 200 Hz on
        material that had no hum at all. The router's rule applies here too:
        doing nothing has to cost nothing.

        A tone is narrow and stands above its own neighbourhood, so it is
        compared against the spectrum a few hertz either side rather than
        against an absolute level — which also makes it independent of how
        loud the recording is.
        """
        spectrum, freqs = self._steady_spectrum(x, rate)
        if spectrum is None:
            return False

        on = (freqs > freq - 2.0) & (freqs < freq + 2.0)
        near = ((freqs > freq - 25.0) & (freqs < freq - 5.0)) | (
            (freqs > freq + 5.0) & (freqs < freq + 25.0)
        )
        if not on.any() or near.sum() < 4:
            return False
        peak = 10 * np.log10(spectrum[on].max() + 1e-30)
        floor = 10 * np.log10(np.median(spectrum[near]) + 1e-30)
        return (peak - floor) > DETECT_DB

    def _steady_spectrum(self, x: np.ndarray, rate: int):
        """The part of the spectrum that is there the whole time.

        A ground loop is constant; a voice is not. Taking the median across
        many one-second frames keeps whatever is always present and suppresses
        anything that comes and goes, so the hum stands out without needing to
        find a pause — which matters, because this material's pauses are often
        gated to exact digital zero and carry no hum either.

        Looking through the speech instead does not work: on material with no
        hum at all, a mean spectrum made the detector fire on 100, 200 and
        300 Hz, because a voice has harmonics there and they stand well above
        their neighbours in any single frame.
        """
        if self._cached is not None and self._cached[0] == x.shape:
            return self._cached[1], self._cached[2]

        window = rate  # 1 s, so bins land on whole hertz
        if len(x) < window * 2:
            return None, None
        starts = np.arange(0, len(x) - window, window // 2)
        taper = np.hanning(window)
        frames = np.array([
            np.abs(np.fft.rfft(x[s:s + window] * taper)) ** 2 for s in starts[:40]
        ])
        if len(frames) < 3:
            return None, None
        spectrum = np.median(frames, axis=0)
        freqs = np.fft.rfftfreq(window, 1 / rate)
        self._cached = (x.shape, spectrum, freqs)
        return spectrum, freqs

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from scipy import signal

        self._cached = None

        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        if len(x) < 32:
            # Too short for a zero-phase filter to have anything to work with.
            return np.asarray(audio, dtype=np.float32).reshape(-1)

        out = x
        self.notched: list[float] = []
        for k in range(1, self.harmonics + 1):
            freq = self.mains_hz * k
            if freq >= rate / 2 * 0.95:
                break
            if not self._present(x, rate, freq):
                continue
            self.notched.append(freq)
            b, a = signal.iirnotch(freq, QUALITY, fs=rate)
            # Zero phase, so the notch does not smear transients in time. The
            # alignment contract makes this mandatory rather than a nicety.
            #
            # The pad has to cover the filter's own settling time, which for a
            # notch this narrow is about 1/(pi * bandwidth) — roughly 0.19 s at
            # 50 Hz with Q of 30. filtfilt's default pad is nine samples, and
            # with that the filter rang at its own centre frequency and *added*
            # 19.7 dB at 50 Hz to material that had no hum in it.
            settle = int(3.0 * rate * QUALITY / max(freq, 1.0))
            pad = int(min(settle, len(out) - 1))
            out = signal.filtfilt(b, a, out, padlen=max(pad, 3 * max(len(a), len(b))))
        return np.ascontiguousarray(out, dtype=np.float32)


@register("notch")
def _load(argument: str) -> Loaded:
    """``notch`` or ``notch:<mains Hz>`` — 50 for Europe, 60 for the US."""
    text = argument.strip()
    if not text:
        return Loaded(NotchEngine())
    try:
        mains = float(text)
    except ValueError:
        raise EngineError(
            f"notch needs a mains frequency as a number, got {text!r} "
            f"— try notch:50 for Europe or notch:60 for the US"
        ) from None
    if not 10.0 < mains < 500.0:
        raise EngineError(f"notch:{text} is not a mains frequency")
    return Loaded(NotchEngine(mains))
