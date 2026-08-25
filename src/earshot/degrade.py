"""Known damage, applied on purpose.

The bench needs pairs: a clean signal and a damaged one that came from it.
Real damaged recordings have no clean counterpart, so the only way to ask
"how much did the engine get back" is to break something good ourselves and
measure the distance to where we started.

Every degradation here is modelled on something that actually happens to a
podcast:

* ``band_limit`` — a remote guest on a narrowband codec. The single most
  common reason a voice sounds like a telephone.
* ``codec`` — Opus/AMR artefacts from a VoIP call. Not the same as band
  limiting: the codec keeps the band and throws away detail inside it.
* ``clip`` — someone leaned into the mic, or an auto-gain stage ran out of
  headroom.
* ``noise`` — a laptop fan, a street, a hissy preamp.
* ``reverb`` — a guest in a kitchen.
* ``dropout`` — packet loss. Short holes, not a level change.
* ``autogain`` — the VoIP platform's own compressor, which is the one
  Sound on Sound reports dxRevive cannot undo. Worth measuring rather
  than assuming.

Degradations compose: a real bad call is band-limited *and* codec-damaged
*and* auto-gained. ``chain`` applies several in the order given, which is the
order they happen in life.

Every function preserves the sample count. The bench compares sample against
sample, and a degradation that shifted the signal would report as damage the
engine could never undo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal


def _sos_lowpass(cutoff: float, rate: int, order: int = 8):
    return signal.butter(order, cutoff / (rate / 2), btype="low", output="sos")


def _sos_highpass(cutoff: float, rate: int, order: int = 4):
    return signal.butter(order, cutoff / (rate / 2), btype="high", output="sos")


def band_limit(x: np.ndarray, rate: int, low: float = 300.0, high: float = 3400.0):
    """The telephone band, or any part of it.

    ``low`` of zero keeps the bottom: a wideband codec throws away the top
    only, and the difference between "sounds like a phone" and "sounds dull"
    is mostly whether the bottom went with it.

    Filtered with ``sosfiltfilt`` so the damage carries no phase shift of its
    own. A restoration engine should not be scored on undoing our filter's
    group delay.
    """
    y = x
    if high and high < rate / 2:
        y = signal.sosfiltfilt(_sos_lowpass(high, rate), y)
    if low and low > 0:
        y = signal.sosfiltfilt(_sos_highpass(low, rate), y)
    return y.astype(np.float32)


def noise(x: np.ndarray, rate: int, snr_db: float = 20.0, seed: int = 0):
    """Broadband noise at a stated signal-to-noise ratio.

    The ratio is measured against the *speech*, not against the whole file:
    a recording that is half silence would otherwise get twice the noise for
    the same nominal SNR.
    """
    active = _speech_rms(x, rate)
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, 1.0, len(x))
    n *= active / (np.sqrt(np.mean(n**2)) + 1e-12) * 10 ** (-snr_db / 20)
    return (x + n).astype(np.float32)


def clip(x: np.ndarray, rate: int, headroom_db: float = -6.0):
    """Hard clipping at a threshold below the signal's own peak."""
    ceiling = np.abs(x).max() * 10 ** (headroom_db / 20)
    return np.clip(x, -ceiling, ceiling).astype(np.float32)


def reverb(x: np.ndarray, rate: int, rt60: float = 0.6, seed: int = 0):
    """A guest in a live room.

    Exponentially decaying noise is not a real room, but it is the standard
    stand-in and it has the property that matters: the tail is uncorrelated
    with the direct sound, so an engine cannot subtract it, only recognise it.
    """
    length = int(rt60 * rate)
    rng = np.random.default_rng(seed)
    ir = rng.normal(0.0, 1.0, length) * np.exp(-np.arange(length) / (rt60 * rate / 6.9))
    ir[0] += 3.0
    # Normalise by energy, not by absolute sum. The L1 version spread the
    # gain over all 28,800 samples of a 0.6 s tail and cost 28.1 dB, so the
    # recipe was moving the speaker twenty times further from the microphone
    # as well as putting them in a live room. Two damages in one recipe means
    # no probe can say which of them an engine failed at. Convolving with an
    # IR of unit energy leaves an uncorrelated signal's level where it was.
    ir /= np.sqrt(np.sum(ir**2))
    return signal.fftconvolve(x, ir)[: len(x)].astype(np.float32)


def dropout(
    x: np.ndarray,
    rate: int,
    rate_per_minute: float = 20.0,
    length_ms: float = 60.0,
    seed: int = 0,
):
    """Packet loss: short holes, tapered so they are gaps and not clicks."""
    y = x.copy()
    count = max(1, int(rate_per_minute * len(x) / rate / 60))
    hole = max(1, int(length_ms * rate / 1000))
    rng = np.random.default_rng(seed)
    ramp = int(min(hole // 4, 0.002 * rate)) or 1
    window = np.ones(hole)
    window[:ramp] = np.linspace(1, 0, ramp)
    window[-ramp:] = np.linspace(0, 1, ramp)
    for start in rng.integers(0, max(1, len(x) - hole), count):
        y[start : start + hole] *= window[: len(y[start : start + hole])]
    return y.astype(np.float32)


def gate(
    x: np.ndarray,
    rate: int,
    threshold_db: float = -18.0,
    hold_ms: float = 80.0,
    release_ms: float = 40.0,
):
    """Silence suppression: the pauses become exact digital zero.

    This is the defect that actually separates remote-platform audio from a
    studio track, and it was the one nothing here modelled. Measured across
    the material in hand: the studio recordings contain no run of eight or
    more zero samples in 224 minutes, while every platform file is 8.0 to
    57.8 per cent exact zero, and the one genuine 7.5 kHz call is 57.8 per
    cent with single gaps reaching 21 seconds.

    It matters more than the added noise the VoIP recipes model, and it runs
    the other way: a denoiser estimates its noise from the pauses, and a gate
    leaves none to estimate from. What survives is speech with hard edges
    into digital silence, which is where a generative engine invents.

    The hold is what stops it chattering inside a word — measured gaps sit in
    the pauses, with 0.0 to 0.3 per cent of them inside speech, so a gate
    that cut into words would model something the material does not do.

    The threshold is relative to the speech level rather than to full scale,
    because the material arrives at every level: speech sits between -27 and
    -52 dBFS across the files measured. -18 dB was chosen by sweeping it on
    two studio tracks until the digital-silence fraction matched the real
    ones: it gives 15.5 to 23.3 per cent against a platform range of 8.0 to
    57.8 per cent and a median near 15.5, while keeping 100 per cent of the
    loud samples. A studio room tone sits only about 22 dB below its speech,
    so a deeper threshold never fires at all — -35 dB gates nothing.
    """
    envelope = _envelope(np.abs(x), rate, attack_ms=1.0, release_ms=release_ms)
    open_gate = envelope > 10 ** (threshold_db / 20) * _speech_rms(x, rate)

    # Hold the gate open for a while after the level drops, so the tail of a
    # word is not clipped off and the gate does not stutter mid-syllable.
    hold = max(1, int(hold_ms * rate / 1000))
    if hold > 1:
        kernel = np.ones(hold, dtype=bool)
        open_gate = np.convolve(open_gate, kernel, mode="same") > 0

    return (x * open_gate).astype(np.float32)


def autogain(
    x: np.ndarray,
    rate: int,
    target_db: float = -20.0,
    attack_ms: float = 5.0,
    release_ms: float = 150.0,
):
    """The VoIP platform's own leveller, pulling everything to one level.

    This is the degradation that removes information rather than adding
    something on top: once the quiet and the loud are the same size, no
    engine can know which was which. Included precisely because it is the
    one restoration is expected to fail at.
    """
    envelope = _envelope(np.abs(x), rate, attack_ms, release_ms)
    target = 10 ** (target_db / 20)
    gain = target / np.maximum(envelope, 1e-5)
    return (x * np.clip(gain, 0.0, 100.0)).astype(np.float32)


def codec(x: np.ndarray, rate: int, bitrate_kbps: int = 16, name: str = "opus"):
    """A real codec round trip through ffmpeg, when ffmpeg is available.

    Simulating codec damage with filters does not work: what a codec does is
    quantise a transform, and the artefacts are the quantisation, not a
    frequency response. When ffmpeg is missing this raises, and the bench
    reports the probe as skipped rather than quietly measuring something else.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    import soundfile as sf

    if not shutil.which("ffmpeg"):
        raise RuntimeError("codec degradation needs ffmpeg on PATH")
    with tempfile.TemporaryDirectory() as work:
        raw = Path(work) / "in.wav"
        squeezed = Path(work) / f"mid.{ 'opus' if name == 'opus' else name }"
        back = Path(work) / "out.wav"
        sf.write(raw, x, rate)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(raw),
             "-b:a", f"{bitrate_kbps}k", str(squeezed)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(squeezed),
             "-ar", str(rate), "-ac", "1", str(back)],
            check=True,
        )
        y, _ = sf.read(back, dtype="float32")
    # Codecs pad and delay. The bench compares sample against sample, so the
    # round trip is realigned and trimmed back to the original length.
    return _align(np.asarray(y).reshape(-1), x).astype(np.float32)


# ------------------------------------------------------------------ helpers


def _speech_rms(x: np.ndarray, rate: int, percentile: float = 75.0) -> float:
    """Level of the loud part, as a stand-in for "the speech"."""
    frame = max(1, int(0.4 * rate))
    count = max(1, len(x) // frame)
    levels = np.array(
        [np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) for i in range(count)]
    )
    loud = levels[levels >= np.percentile(levels, percentile)]
    return float(loud.mean() if len(loud) else np.sqrt(np.mean(x**2)))


def _envelope(x: np.ndarray, rate: int, attack_ms: float, release_ms: float):
    attack = np.exp(-1.0 / (attack_ms * rate / 1000))
    release = np.exp(-1.0 / (release_ms * rate / 1000))
    # A one-pole follower is a serial recurrence; over a whole file in Python
    # it is slow enough to matter, so it runs on a decimated envelope and is
    # interpolated back. The follower's own time constants are milliseconds,
    # far longer than the decimation step.
    step = max(1, rate // 1000)
    coarse = np.maximum.reduceat(x, np.arange(0, len(x), step))
    out = np.empty_like(coarse)
    level = 0.0
    a = attack ** step
    r = release ** step
    for i, value in enumerate(coarse):
        coeff = a if value > level else r
        level = coeff * level + (1 - coeff) * value
        out[i] = level
    return np.interp(np.arange(len(x)), np.arange(len(coarse)) * step, out)


def _align(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Shift ``y`` onto ``reference`` and match its length."""
    n = len(reference)
    probe = min(len(y), n, 200000)
    spectrum = np.fft.rfft(y[:probe], probe * 2) * np.conj(
        np.fft.rfft(reference[:probe], probe * 2)
    )
    lag = int(np.argmax(np.fft.irfft(spectrum)))
    if lag > probe:
        lag -= probe * 2
    y = y[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, dtype=y.dtype), y])
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y), dtype=y.dtype)])
    return y[:n]


# --------------------------------------------------------------- the recipes


@dataclass(frozen=True)
class Damage:
    """One named degradation, ready to apply.

    ``needs`` names an external tool the recipe cannot work without, so the
    bench can report "skipped, no ffmpeg" instead of failing or, worse,
    silently measuring a different thing.
    """

    name: str
    describe: str
    steps: tuple[tuple[Callable, dict], ...] = field(default_factory=tuple)
    needs: str = ""

    def apply(self, x: np.ndarray, rate: int) -> np.ndarray:
        for function, options in self.steps:
            x = function(x, rate, **options)
        return x


RECIPES: tuple[Damage, ...] = (
    Damage(
        "clean",
        "untouched — the control, and the only honest way to see what an "
        "engine does to material that needs nothing",
    ),
    Damage(
        "hiss",
        "broadband noise at 20 dB SNR: a fan, a preamp, a street",
        ((noise, {"snr_db": 20.0}),),
    ),
    Damage(
        "room",
        "a live room, RT60 0.6 s",
        ((reverb, {"rt60": 0.6}),),
    ),
    Damage(
        "wideband-voip",
        "a decent call: 8 kHz ceiling, mild noise, a little auto-gain",
        (
            # The noise goes in before the band limit, not after. A call
            # carries the room and the microphone's own noise, and then the
            # codec band-limits all of it; nothing above the ceiling is ever
            # transmitted. Adding broadband noise afterwards refilled the
            # band this recipe exists to empty — measured at -24 dB below the
            # speech band where band_limit had left -147 dB — so the recipe
            # stopped modelling a call, the router declined to engage on a
            # signal that really did reach 24 kHz, and an engine scored for
            # removing noise no telephone band would have carried.
            (noise, {"snr_db": 30.0}),
            (band_limit, {"low": 80.0, "high": 8000.0}),
            (autogain, {"target_db": -20.0}),
        ),
    ),
    Damage(
        "platform-upload",
        "what a remote-recording platform delivers: 15 kHz ceiling, gated silence",
        (
            # Built from measurement rather than from what a call is imagined
            # to do. Across sixteen files: every remote-platform track is
            # band-limited to 14.0-15.6 kHz with a flat stopband, and 8.0 to
            # 57.8 per cent of it is exact digital zero, while the studio
            # tracks reach 24 kHz and hold no run of eight zero samples in
            # 224 minutes. Packet loss inside speech is 0.0 to 0.3 per cent
            # and clipping single-digit parts per million, so neither is
            # modelled here — and no noise is added, because the platform
            # removes the room tone rather than adding to it.
            (gate, {}),
            (band_limit, {"low": 0.0, "high": 15000.0}),
        ),
    ),
    Damage(
        "narrowband-voip",
        "the bad call: telephone band, clipping, packet loss, auto-gain",
        (
            (band_limit, {"low": 300.0, "high": 3400.0}),
            (clip, {"headroom_db": -8.0}),
            (dropout, {"rate_per_minute": 20.0}),
            (autogain, {"target_db": -20.0}),
        ),
    ),
    Damage(
        "opus-16k",
        "a real Opus round trip at 16 kbit/s",
        ((codec, {"bitrate_kbps": 16}),),
        needs="ffmpeg",
    ),
)


def by_name(name: str) -> Damage:
    for recipe in RECIPES:
        if recipe.name == name:
            return recipe
    known = ", ".join(r.name for r in RECIPES)
    raise KeyError(f"unknown damage {name!r}; known: {known}")
