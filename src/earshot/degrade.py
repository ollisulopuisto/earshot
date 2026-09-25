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


def hum(
    x: np.ndarray,
    rate: int,
    level_db: float = -30.0,
    mains_hz: float = 50.0,
    harmonics: int = 6,
    drift_hz: float = 0.05,
    seed: int = 0,
):
    """A ground loop: a mains fundamental and its harmonics, and nothing else.

    The bench modelled broadband hiss and nothing narrowband, while a buzz
    from a ground loop is the complaint that actually reaches a producer. It
    is a different problem for an engine: a wideband denoiser has to notice
    that a handful of bins are the enemy, and a notch filter that knows the
    mains frequency removes it outright.

    ``level_db`` is relative to the speech, not to full scale, because the
    material arrives at every level. The default of -30 dB is an audible
    ground loop rather than a measurement of this project's material: in the
    owner's own recordings the 50 Hz family sits 32 to 51 dB below speech,
    which is at or below the room tone and inaudible. Calibrating to that
    would have modelled a problem nobody has.

    Mains frequency drifts by a few hundredths of a hertz as grid load
    changes, which is why a fixed notch leaves a residue and why the drift is
    modelled rather than assumed away. 50 Hz is Europe; pass 60 for the US.
    """
    n = len(x)
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)

    # A slow random walk in frequency, integrated into phase.
    if drift_hz > 0:
        steps = rng.normal(0.0, 1.0, n // rate + 2)
        walk = np.interp(t, np.arange(len(steps)) * 1.0, np.cumsum(steps))
        walk = drift_hz * walk / (np.abs(walk).max() + 1e-12)
    else:
        walk = np.zeros(n)

    buzz = np.zeros(n)
    for k in range(1, harmonics + 1):
        # Harmonics fall away, and the odd ones sit higher: that is what a
        # transformer-coupled loop looks like on an analyser.
        weight = (1.0 / k) * (1.0 if k % 2 else 0.6)
        phase = 2 * np.pi * (mains_hz * k * t + k * np.cumsum(walk) / rate)
        buzz += weight * np.sin(phase + rng.uniform(0, 2 * np.pi))

    buzz /= np.sqrt(np.mean(buzz**2)) + 1e-12
    buzz *= _speech_rms(x, rate) * 10 ** (level_db / 20)
    return (x + buzz).astype(np.float32)


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


def mains(
    x: np.ndarray,
    rate: int,
    mains_hz: float = 50.0,
    level_db: float = -30.0,
    harmonics: int = 6,
    slope_db: float = -6.0,
    odd_only: bool = False,
    drift_hz: float = 0.15,
    seed: int = 0,
):
    """Mains interference with a controllable spectrum, for hum and for buzz.

    Written by a later session that did not know ``hum()`` existed. ``hum()``
    is a ground loop with a fixed 1/k spectrum and a random-walk drift, and
    its results were measured on the owner's own material; it is kept exactly
    as it was so those numbers still mean what they say. This one adds what
    ``hum()`` does not: a spectral slope and an odd-only mode, which a
    rectifier's buzz reaching several kHz needs, and a larger drift.

    Not measured on this project's material yet — no hum has been found in
    the studio tracks, and the same-room recordings where a spec expects it
    have not been through the bench. The shape is the textbook one: a
    ground loop gives mostly 50 Hz and its low harmonics; a switched supply
    or a rectifier gives a buzz whose harmonics reach several kHz, stronger
    on the odd ones. European mains wander by a few tenths of a hertz, which
    is what makes a fixed notch comb miss and why ``drift_hz`` is here.

    ``level_db`` is the fundamental relative to the speech level, so the
    same recipe is equally audible on a hot file and a quiet one. Harmonic
    *n* sits ``slope_db`` per octave below it.
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    # A slow wander of the mains frequency: one full swing over roughly
    # half a minute, which is the order of what grid frequency does.
    wobble = drift_hz * np.sin(2 * np.pi * t / 27.0 + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(mains_hz + wobble) / rate
    speech = _speech_rms(x, rate)
    base = speech * 10 ** (level_db / 20) * np.sqrt(2)
    y = np.zeros(n)
    for k in range(1, harmonics + 1):
        if odd_only and k > 1 and k % 2 == 0:
            continue
        if k * mains_hz >= rate / 2 * 0.9:
            break
        amplitude = base * 10 ** (slope_db * np.log2(k) / 20)
        y += amplitude * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    return (x + y).astype(np.float32)


def plosive(
    x: np.ndarray,
    rate: int,
    per_minute: float = 12.0,
    level_db: float = 0.0,
    seed: int = 0,
):
    """Breath hitting the capsule at the start of a word: a low thump.

    A plosive pop is a pressure wave, not a sound: most of its energy sits
    below 150 Hz, it lasts a few tens of milliseconds, and it lands on a
    speech onset — a /p/ or /b/ — rather than at random. Its peak can exceed
    the speech peak, which is why it clips as often as it thumps.

    Modelled as a damped low-frequency swing starting at onsets picked from
    the speech envelope: a rise of a few milliseconds, then a decay of
    30–80 ms at 25–70 Hz. Not fitted to measured pops yet; the podcast
    archive has not been searched for them. ``level_db`` is the pop's peak
    relative to the speech's own 99.9th-percentile peak.
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    y = x.astype(np.float64).copy()
    envelope = _envelope(np.abs(x), rate, attack_ms=2.0, release_ms=60.0)
    # The envelope follows peaks, so it is compared against its own loud
    # level rather than against an RMS: against a quarter of the speech RMS,
    # a fluent speaker was "loud" in 99.997 per cent of samples and had no
    # onsets at all.
    loud = envelope > 0.2 * np.percentile(envelope, 90)
    # Onsets: where the envelope crosses into speech after at least 60 ms
    # below it, so the pop lands on the start of a word. Longer rests found
    # no onsets at all in twelve seconds of a fluent speaker.
    rest = int(0.06 * rate)
    counts = np.concatenate(([0], np.cumsum(~loud)))
    quiet_run = (counts[1:] - counts[np.maximum(0, np.arange(1, n + 1) - rest)]) >= rest
    onsets = np.flatnonzero(loud[1:] & ~loud[:-1] & quiet_run[:-1]) + 1
    if not len(onsets):
        return y.astype(np.float32)
    wanted = max(1, int(round(per_minute * n / rate / 60)))
    picked = rng.choice(onsets, size=min(wanted, len(onsets)), replace=False)
    peak = float(np.percentile(np.abs(x), 99.9)) * 10 ** (level_db / 20)
    for start in np.sort(picked):
        # A few ms before the vowel: the release of the stop, not the vowel.
        start = max(0, int(start - rng.uniform(0.0, 0.01) * rate))
        decay = rng.uniform(0.03, 0.08)
        frequency = rng.uniform(25.0, 70.0)
        length = min(n - start, int(decay * 5 * rate))
        if length <= 0:
            continue
        t = np.arange(length) / rate
        shape = (1 - np.exp(-t / 0.003)) * np.exp(-t / decay)
        burst = shape * np.sin(2 * np.pi * frequency * t)
        burst *= peak / (np.max(np.abs(burst)) + 1e-12)
        y[start : start + length] += burst * rng.choice((-1.0, 1.0))
    return y.astype(np.float32)


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
        "ground-loop",
        "a mains buzz: 50 Hz and its harmonics, 30 dB under the speech",
        ((hum, {}),),
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
        "hum",
        "mains hum: 50 Hz and five harmonics, -30 dB under the speech, drifting",
        ((mains, {"mains_hz": 50.0, "level_db": -30.0, "harmonics": 6}),),
    ),
    Damage(
        "buzz",
        "supply buzz: odd-heavy 50 Hz harmonics up to 4 kHz, -36 dB",
        (
            # Harmonics to 4 kHz at -3 dB per octave: a rectifier's spike
            # train is broadband, which is what separates buzz from hum and
            # what defeats a notch comb that stops at the fifth harmonic.
            (mains, {"mains_hz": 50.0, "level_db": -36.0, "harmonics": 80,
                   "slope_db": -3.0, "odd_only": True}),
        ),
    ),
    Damage(
        "plosive",
        "breath pops on word onsets: 12 a minute, peaking at the speech peak",
        ((plosive, {"per_minute": 12.0, "level_db": 0.0}),),
    ),
    Damage(
        "clipped",
        "hard clipping 6 dB under the peak, and nothing else",
        ((clip, {"headroom_db": -6.0}),),
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
