"""What "better" means, in numbers.

Every metric here answers a question someone would otherwise settle by
arguing. They fall into two groups, and the difference matters:

**Referenced** metrics compare against the clean original. They are only
available for material the bench damaged itself, and they are the ones that
can say "the engine got 1.5 dB of the missing band back".

**Unreferenced** metrics work on any recording, including a real bad call
with no clean counterpart. They cannot say whether the result is *right*,
only whether it has the properties a restored voice should have: a lower
noise floor, speech left at its own level, and non-speech not deleted.

The numbers here were calibrated against a commercial restoration plug-in
(Accentize dxRevive 1.1.1) on real podcast material — see ``docs/dxrevive.md``.
That is what the thresholds in ``probes.py`` are relative to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x) ** 2)) + 1e-12))


def log_spectral_distance(
    a: np.ndarray,
    b: np.ndarray,
    rate: int,
    low: float = 0.0,
    high: float = 0.0,
    match_level: bool = True,
) -> float:
    """Distance between two magnitude spectra, in dB, over a frequency band.

    The workhorse for bandwidth extension: restrict the band to the part the
    damage removed and the number says how close the engine got to the
    content that used to be there. Lower is better; zero is identical.

    Averaged over frames, not over the whole file, so a loud passage cannot
    outvote a quiet one.

    ``match_level`` is on because it has to be. A pure gain change shifts
    every bin by the same number of decibels, so without it this metric
    scores "the same spectrum, louder" as damage — and every engine that
    normalises its output, which is most of them, would be punished for
    something inaudible. The first run of this bench against a real plug-in
    reported it as *losing* 1.5 dB where a hand measurement said it gained
    1.5 dB, and this was the whole difference.

    Note what level matching cannot fix: a degradation that destroys
    dynamics rather than applying a constant gain — ``degrade.autogain`` —
    leaves a mismatch no single scale factor removes. That is not a flaw in
    the metric. It is the reason auto-gain is the damage nobody can undo,
    and the bench should show it.
    """
    n = min(len(a), len(b))
    x, y = np.asarray(a)[:n], np.asarray(b)[:n]
    if match_level:
        scale = (np.sqrt(np.mean(y**2)) + 1e-12) / (np.sqrt(np.mean(x**2)) + 1e-12)
        x = x * scale
    _, _, A = signal.stft(x, rate, nperseg=1024)
    _, _, B = signal.stft(y, rate, nperseg=1024)
    freqs = np.fft.rfftfreq(1024, 1 / rate)
    mask = np.ones(len(freqs), dtype=bool)
    if high:
        mask &= freqs < high
    if low:
        mask &= freqs >= low
    if not mask.any():
        return float("nan")
    la = 10 * np.log10(np.abs(A[mask]) ** 2 + 1e-10)
    lb = 10 * np.log10(np.abs(B[mask]) ** 2 + 1e-10)
    return float(np.sqrt(((la - lb) ** 2).mean()))


def band_correlation(
    a: np.ndarray, b: np.ndarray, rate: int, low: float, high: float
) -> float:
    """How much of the output is still the input, in one band.

    This is the metric that tells filtering from invention, and it is the
    one no listening test gives you. A spectral mask leaves the waveform's
    phase intact and correlates near 1. A vocoder that resynthesises the
    band correlates near 0 even when it sounds excellent.

    Neither is wrong. But an engine that scores 0.1 where the input was
    perfectly good has replaced the speaker's voice with its idea of a
    voice, and that is worth knowing before it ships.
    """
    n = min(len(a), len(b))
    nyquist = rate / 2
    high = min(high, nyquist * 0.99)
    if low >= high:
        return float("nan")
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    fa = signal.sosfilt(sos, np.asarray(a)[:n])
    fb = signal.sosfilt(sos, np.asarray(b)[:n])
    if np.std(fa) < 1e-12 or np.std(fb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(fa, fb)[0, 1])


@dataclass(frozen=True)
class Dynamics:
    """Where the speech sits and where the silence sits, in dBFS."""

    speech: float
    floor: float

    @property
    def range_db(self) -> float:
        return self.speech - self.floor


def dynamics(x: np.ndarray, rate: int, fraction: float = 0.2) -> Dynamics:
    """Level of the loudest and quietest fifths, by 400 ms frame.

    The pair matters more than either number. A denoiser that lowers the
    floor by 10 dB and the speech by 10 dB has done nothing but turn down;
    one that lowers the floor and leaves the speech is what "cleaner" means.
    """
    frame = max(1, int(0.4 * rate))
    count = max(1, len(x) // frame)
    levels = np.array([rms_db(x[i * frame : (i + 1) * frame]) for i in range(count)])
    take = max(1, int(count * fraction))
    order = np.argsort(levels)
    return Dynamics(
        speech=float(levels[order[-take:]].mean()),
        floor=float(levels[order[:take]].mean()),
    )


def suppression(before: np.ndarray, after: np.ndarray) -> float:
    """How much quieter the output is than the input, in dB.

    Used on signals that are not speech. A speech-only model treats anything
    else as noise and removes it, which is correct for a lone voice and
    destructive for a recording with a room, music, or a second sound in it.
    """
    return rms_db(before) - rms_db(np.asarray(after)[: len(before)])


def spectral_tilt(x: np.ndarray, rate: int, low: float = 200.0, high: float = 8000.0):
    """Slope of the average spectrum in dB per octave.

    A crude but stable measure of "dull" versus "bright". Restoration that
    only adds a high shelf shows up here and nowhere else.
    """
    freqs, power = signal.welch(np.asarray(x), rate, nperseg=4096)
    mask = (freqs >= low) & (freqs <= min(high, rate / 2 * 0.99)) & (power > 0)
    if mask.sum() < 8:
        return float("nan")
    octaves = np.log2(freqs[mask] / low)
    decibels = 10 * np.log10(power[mask])
    return float(np.polyfit(octaves, decibels, 1)[0])


def repeatability(first: np.ndarray, second: np.ndarray) -> float:
    """How far below the signal the difference between two runs sits, in dB.

    A number rather than a yes/no, because the answer turned out not to be
    binary. A commercial plug-in measured here is not bit-exact between runs
    but repeats to about -60 dB — inaudible, yet enough to fail any equality
    test. Reporting the margin says both things at once: it is stable enough
    to trust, and it is not a pure function.

    Higher is better. Above ~120 dB means bit-identical in float32.
    """
    n = min(len(first), len(second))
    difference = rms_db(np.asarray(first)[:n] - np.asarray(second)[:n])
    return float(rms_db(np.asarray(first)[:n]) - difference)


# --------------------------------------------------------------- perceptual

# PESQ and STOI are the two numbers every speech restoration paper reports,
# and the only reason they are here is comparability: a claim of "3.1 PESQ"
# in a paper means nothing next to our own metrics unless we can compute the
# same figure on the same material.
#
# Both are referenced — they need the clean original — so they say nothing
# about a real bad recording. And both were designed for narrowband telephony
# and codec artefacts, which is exactly our VoIP case and exactly not our
# good-microphone case. Read them as "how would the literature score this",
# not as "how good does it sound".
PESQ_RATE = 16000


def _resampled(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return np.asarray(x, dtype=np.float64)
    from math import gcd

    factor = gcd(int(rate), int(target))
    return signal.resample_poly(
        np.asarray(x, dtype=np.float64), target // factor, rate // factor
    )


def perceptual_available() -> bool:
    try:
        import pesq  # noqa: F401
        import pystoi  # noqa: F401
    except ImportError:
        return False
    return True


def perceptual(reference: np.ndarray, degraded: np.ndarray, rate: int) -> dict:
    """PESQ (wideband MOS-LQO, 1–4.5) and STOI (intelligibility, 0–1).

    Returns an empty dict when the optional dependencies are absent, so the
    bench reports nothing rather than reporting a substitute. A metric that
    silently falls back to something else is worse than a missing one.

    Both are computed at 16 kHz: PESQ accepts nothing else in wideband mode,
    and STOI resamples internally anyway, so doing it once keeps the two
    reading the same signal.
    """
    if not perceptual_available():
        return {}
    from pesq import pesq as _pesq
    from pystoi import stoi as _stoi

    n = min(len(reference), len(degraded))
    ref = _resampled(np.asarray(reference)[:n], rate, PESQ_RATE)
    deg = _resampled(np.asarray(degraded)[:n], rate, PESQ_RATE)
    out: dict = {}
    try:
        out["pesq"] = float(_pesq(PESQ_RATE, ref, deg, "wb"))
    except Exception:
        # PESQ refuses signals with no active speech, which happens on a
        # silent excerpt. That is a fact about the material, not an error.
        pass
    try:
        out["stoi"] = float(_stoi(ref, deg, PESQ_RATE, extended=False))
    except Exception:
        pass
    return out
