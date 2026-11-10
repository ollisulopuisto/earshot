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
from functools import lru_cache

import numpy as np
from scipy import signal


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x) ** 2)) + 1e-12))


class LogSpectralDistancePlan:
    """Reference spectrum and band mask shared by aligned distance scores.

    The plan owns only derived arrays and parameters, not the caller's reference.
    Its lifetime is one recovery probe, so before/after reuse does not retain
    recordings globally.
    """

    def __init__(
        self,
        reference: np.ndarray,
        rate: int,
        low: float = 0.0,
        high: float = 0.0,
        match_level: bool = True,
    ):
        r = np.asarray(reference)
        if r.ndim != 1:
            raise ValueError("reference must be a one-dimensional signal")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be positive and finite")
        self._length = len(r)
        self._rate = rate
        self._match_level = match_level
        self._ref_rms = float(np.sqrt(np.mean(r**2)) + 1e-12) if len(r) > 0 else 0.0

        if len(r) == 0:
            self._valid = False
            self._lb = None
            self._mask = np.zeros(0, dtype=bool)
            return

        freqs = np.fft.rfftfreq(1024, 1 / rate)
        mask = np.ones(len(freqs), dtype=bool)
        if high:
            mask &= freqs < high
        if low:
            mask &= freqs >= low
        self._mask = mask

        if not mask.any():
            self._valid = False
            self._lb = None
        else:
            self._valid = True
            _, _, B = signal.stft(r, rate, nperseg=1024)
            self._lb = 10 * np.log10(np.abs(B[mask]) ** 2 + 1e-10)

    def score(self, x: np.ndarray) -> float:
        """Score an audio signal with the reference's exact shape and sample alignment."""
        # Alignment is the contract the whole bench rests on, so it is
        # checked before anything can return: an empty band makes the score
        # NaN, but it must not let a misaligned signal slip the shape guard.
        arr = np.asarray(x)
        if arr.shape != (self._length,):
            raise ValueError(
                f"signal shape {arr.shape} must match reference shape ({self._length},)"
            )
        if not self._valid:
            return float("nan")
        if self._match_level:
            scale = self._ref_rms / (np.sqrt(np.mean(arr**2)) + 1e-12)
            arr = arr * scale
        _, _, A = signal.stft(arr, self._rate, nperseg=1024)
        la = 10 * np.log10(np.abs(A[self._mask]) ** 2 + 1e-10)
        return float(np.sqrt(((la - self._lb) ** 2).mean()))


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
    # No samples means nothing to measure. NaN, not a raise and not zero:
    # zero would read as a perfect restoration of an absent signal.
    if n == 0:
        return float("nan")
    return LogSpectralDistancePlan(
        np.asarray(b)[:n], rate, low=low, high=high, match_level=match_level
    ).score(np.asarray(a)[:n])


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
    arr = np.asarray(x)
    if len(arr) >= frame:
        truncated = arr[: count * frame].reshape(count, frame)
        means = np.mean(truncated**2, axis=1)
        levels = 20 * np.log10(np.sqrt(means).astype(np.float64) + 1e-12)
    else:
        levels = np.array([rms_db(arr)])
    take = max(1, int(count * fraction))
    order = np.argsort(levels)
    return Dynamics(
        speech=float(levels[order[-take:]].mean()),
        floor=float(levels[order[:take]].mean()),
    )


class DynamicsMeter:
    """``dynamics()`` over a signal that arrives in blocks.

    Streaming restore cannot hand ``dynamics()`` the whole recording - not
    holding it is the point. Frames are counted from the start of the
    stream, not of any block, and only complete frames count, so
    ``result()`` equals ``dynamics()`` over the whole signal, partial tail
    frame dropped the same way.
    """

    def __init__(self, rate: int, fraction: float = 0.2):
        self._frame = max(1, int(0.4 * rate))
        self._fraction = fraction
        self._carry = np.empty(0, dtype=np.float32)
        self._levels: list[float] = []

    def add(self, x: np.ndarray) -> None:
        x = np.concatenate([self._carry, np.asarray(x).reshape(-1)])
        whole = len(x) // self._frame * self._frame
        for i in range(0, whole, self._frame):
            self._levels.append(rms_db(x[i : i + self._frame]))
        self._carry = x[whole:]

    def result(self) -> Dynamics:
        if self._levels:
            levels = np.array(self._levels)
        else:
            # Shorter than one frame: dynamics() scores the whole signal as
            # a single short frame, so the carry - everything seen - is it.
            levels = np.array([rms_db(self._carry)])
        take = max(1, int(len(levels) * self._fraction))
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


BODY_BAND_HZ = (80.0, 250.0)
SPEECH_BAND_HZ = (300.0, 3000.0)


@lru_cache(maxsize=16)
def _body_welch_plan(
    nperseg: int, rate: int
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    # Cache Hann window and band bin slices: avoids re-evaluating window design
    # and linear boolean mask scans across ~4097 bins on every metric call.
    win = signal.windows.get_window("hann", nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1 / rate)
    low_slice = (
        int(np.searchsorted(freqs, BODY_BAND_HZ[0])),
        int(np.searchsorted(freqs, BODY_BAND_HZ[1])),
    )
    mid_slice = (
        int(np.searchsorted(freqs, SPEECH_BAND_HZ[0])),
        int(np.searchsorted(freqs, SPEECH_BAND_HZ[1])),
    )
    return win, low_slice, mid_slice


def body(x: np.ndarray, rate: int) -> float:
    """The level of 80–250 Hz against 300–3000 Hz, in dB.

    Where a voice's fundamental and second harmonic live, against where its
    intelligibility does: a balance, so a fader move leaves it alone. A
    telephone band cuts at 300 Hz and takes this about 20 dB down (measured
    on the listening sets, 2026-10-03), which is the thinness the owner heard.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 256:
        return float("nan")
    nperseg = min(8192, len(x))
    win, (i_lo1, i_hi1), (i_lo2, i_hi2) = _body_welch_plan(nperseg, rate)

    # Welch periodogram accumulation directly into target bands:
    # Avoids scipy.signal.welch/ShortTimeFFT overhead (allocating full
    # 2D spectrograms, computing unused bins above 3 kHz, and scaling factors
    # that cancel in the decibel ratio anyway).
    hop = nperseg // 2
    n_seg = max(1, (len(x) - hop) // hop)
    low_sum = 0.0
    mid_sum = 0.0
    for p in range(n_seg):
        seg = x[p * hop : p * hop + nperseg]
        fft_res = np.fft.rfft((seg - seg.mean()) * win)
        pwr = fft_res.real**2 + fft_res.imag**2
        low_sum += pwr[i_lo1:i_hi1].sum()
        mid_sum += pwr[i_lo2:i_hi2].sum()

    return float(10 * np.log10((low_sum + 1e-30) / (mid_sum + 1e-30)))


def speaker_similarity(a: np.ndarray, b: np.ndarray, rate: int) -> float:
    """Cosine between speaker embeddings of ``a`` and ``b``; 1 is one voice.

    Raises EngineError when the speaker model is unavailable, so a caller
    reports a skip rather than a number about nothing.
    """
    from . import speaker

    return float(np.dot(speaker.embed(a, rate), speaker.embed(b, rate)))


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


class LineResiduePlan:
    """Reference spectrum and harmonic mask shared by aligned error scores.

    The plan owns only derived arrays, not the caller's reference. Its lifetime
    is one tonal probe, so before/after reuse does not retain recordings globally.
    """

    def __init__(self, reference: np.ndarray, rate: int,
                 fundamental: float = 50.0, top: float = 6000.0):
        r = np.asarray(reference, dtype=np.float64)
        if r.ndim != 1 or not len(r):
            raise ValueError("reference must be a nonempty one-dimensional signal")
        if not np.isfinite(fundamental) or fundamental <= 0:
            raise ValueError("fundamental must be finite and positive")
        if not np.isfinite(rate) or rate <= 0 or not np.isfinite(top):
            raise ValueError("rate must be positive and rate/top must be finite")
        self._window = np.hanning(len(r))
        freqs = np.fft.rfftfreq(len(r), 1 / rate)
        lines = np.zeros_like(freqs, dtype=bool)
        # Narrow-slice window evaluation: because freqs is strictly increasing,
        # find each harmonic window's index range via searchsorted and evaluate
        # the condition on that slice instead of scanning all bins for every k.
        limit = min(top, rate / 2 * 0.95)
        k_max = int(limit // fundamental)
        if k_max >= 1:
            ks = np.arange(1, k_max + 1)
            ks = ks[ks * fundamental < limit]
            centres = ks * fundamental
            widths = 1.0 + 0.3 * ks
            idx_lo = np.searchsorted(freqs, centres - widths, side="left")
            idx_hi = np.searchsorted(freqs, centres + widths, side="right")
            for c, w, lo_i, hi_i in zip(centres, widths, idx_lo, idx_hi):
                lines[lo_i:hi_i] |= np.abs(freqs[lo_i:hi_i] - c) <= w
        self._lines = lines
        self._speech = np.sum(np.abs(np.fft.rfft(r * self._window)) ** 2) + 1e-30

    def score(self, error: np.ndarray) -> float:
        """Score an error with the reference's exact shape and sample alignment."""
        e = np.asarray(error, dtype=np.float64)
        if e.shape != self._window.shape:
            raise ValueError("error shape must match the reference shape")
        spectrum = np.abs(np.fft.rfft(e * self._window)) ** 2
        return float(10 * np.log10(spectrum[self._lines].sum() / self._speech + 1e-30))


def line_residue(error: np.ndarray, reference: np.ndarray, rate: int,
                 fundamental: float = 50.0, top: float = 6000.0) -> float:
    """Energy of ``error`` at the harmonics of ``fundamental``, in dB of speech.

    Each harmonic's window widens with its number, because drift of a few
    tenths of a hertz at the fundamental is k times that at the k-th
    harmonic. What falls between the harmonics is not counted: that is the
    voice, and other probes answer for it.
    """
    return LineResiduePlan(reference, rate, fundamental, top).score(error)
