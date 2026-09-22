"""Mains hum and buzz, removed by subtraction rather than by filtering.

    earshot bench --engine dehum          # finds 50 or 60 Hz itself
    earshot bench --engine dehum:50       # European mains only
    earshot bench --engine dehum:50@2     # a wider tracking bandwidth, in Hz

Hum is the one damage here that needs no model at all: it is a sum of
sinusoids whose frequencies are known to within a hertz before the file is
opened. What it needs is care, because the obvious tool — a comb of fixed
notches — fails in two ways. The mains frequency drifts by tenths of a hertz,
so a notch narrow enough to spare the voice misses the twentieth harmonic;
and a notch wide enough to catch it takes a slice out of every voice whose
pitch passes through.

**How.** The fundamental is tracked, block by block, by fitting the
harmonics' combined energy over a fine grid around the nominal frequency.
Each harmonic is then *demodulated*: multiplied down to 0 Hz by the tracked
phase, averaged over about a second, and multiplied back up. What survives
the averaging is whatever sat still at that frequency for a second — hum —
while a voice passing through the same frequency moves on long before the
average notices it. The estimate is subtracted. Nothing below, between or
above the harmonics is touched, and the phase of the voice is untouched
everywhere.

**Which harmonics.** Only those that stand clear of their neighbourhood.
A harmonic is treated if its steady level beats the level measured the same
way a few hertz either side by ``PROMINENCE_DB``. That is what keeps the
engine from subtracting anything from a recording that has no hum — the
first thing the `clean` recipe will check.

A guess at the right defaults, not a measured optimum: no hum has yet been
found in this project's own material, so the recipe it is tested against is
the textbook one.
"""

from __future__ import annotations

import numpy as np

from . import EngineError, Loaded, register

# Tracking and averaging happen on a decimated baseband. 10 ms blocks are far
# shorter than the averaging window, so the decimation costs nothing audible.
BLOCK_S = 0.01

# Bandwidth of the averaging, in Hz: the width of each notch. One hertz
# spares everything a voice does and follows a drift of half a hertz a
# second at the fundamental.
DEFAULT_BANDWIDTH_HZ = 1.0

# How far a harmonic must stand above its neighbourhood to be treated. Six
# decibels is a guess: loud enough that speech energy, which is spread, does
# not qualify on its own.
PROMINENCE_DB = 6.0

# The second test, per harmonic on the tracked baseband, after the survey.
# Lowered to 1 dB it let four harmonics through on a recording with no hum.
LOCAL_PROMINENCE_DB = 6.0

# Harmonics are sought up to here. Buzz reaches several kHz; beyond this
# the voice dominates and a mistake costs more than a residue.
TOP_HZ = 6000.0

# The fundamental is re-estimated over windows this long.
TRACK_S = 2.0

# Search range around the nominal mains frequency.
SEARCH_HZ = 1.0


def _moving_average(z: np.ndarray, length: int) -> np.ndarray:
    """Centred boxcar of odd length, edges averaged over what exists."""
    half = length // 2
    c = np.concatenate(([0], np.cumsum(z)))
    idx = np.arange(len(z))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(z), idx + half + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def _blocks(shifted: np.ndarray, block: int) -> np.ndarray:
    """A signal already moved to 0 Hz, lowpassed and averaged into blocks.

    A single block average is not enough, and measured to matter: its
    response has sidelobes, and decimating to 100 Hz folds a voice's
    harmonic lying a hertz or two from 150, 250, 350 Hz onto the 50 Hz
    estimate. On a low male voice that folded speech was as loud as the hum,
    and subtracting the estimate *added* 2.7 dB of error rather than taking
    the hum out. Three boxcars in cascade (a CIC filter) push the folded
    region down by the cube of one sidelobe instead.
    """
    odd = block | 1
    # Reflected at the ends, so the first and last blocks are averaged over
    # a full window too. Truncated windows reject a voice far less well, and
    # the smoothing after this extends the ends by odd reflection, so one
    # contaminated edge block was amplified into -49 dB of error in the first
    # half-second of every excerpt, louder than the hum it was removing.
    edge = min(2 * odd, len(shifted) - 1)
    padded = np.pad(shifted, edge, mode="reflect") if edge > 0 else shifted
    padded = _moving_average(_moving_average(padded, odd), odd)
    shifted = padded[edge : edge + len(shifted)]
    whole = len(shifted) // block * block
    head = shifted[:whole].reshape(-1, block).mean(axis=1)
    if whole < len(shifted):
        head = np.append(head, shifted[whole:].mean())
    return head


def _smooth(z: np.ndarray, bandwidth_hz: float, block_rate: float) -> np.ndarray:
    """Zero-phase lowpass of a complex baseband, so the estimate has no lag."""
    from scipy import signal

    cutoff = bandwidth_hz / 2
    if len(z) < 16 or cutoff >= block_rate / 2:
        return np.full_like(z, z.mean())
    sos = signal.butter(2, cutoff / (block_rate / 2), output="sos")
    pad = min(len(z) - 1, int(3 * block_rate / cutoff))
    return (signal.sosfiltfilt(sos, z.real, padlen=pad, padtype="even")
            + 1j * signal.sosfiltfilt(sos, z.imag, padlen=pad, padtype="even"))


class DehumEngine:
    def __init__(self, nominal: float | None = None,
                 bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ):
        self.nominal = nominal
        self.bandwidth_hz = bandwidth_hz
        label = f"{nominal:g}" if nominal else "auto"
        self.name = (f"dehum({label})" if bandwidth_hz == DEFAULT_BANDWIDTH_HZ
                     else f"dehum({label}@{bandwidth_hz:g}Hz)")
        self.found: dict = {}

    # --------------------------------------------------------------- survey

    def _survey(self, x: np.ndarray, rate: int, nominal: float) -> list[int]:
        """Harmonics that stand out of the long-term spectrum.

        Done first, and cheaply, because everything after depends on it. The
        first version tracked the mains frequency on the first eight
        harmonics regardless, and on a low male voice harmonics three to
        eight sit *under* the speech: measured on one, the hum there was
        0.5 to 1.5 dB above the voice's own spectrum. Their contribution to
        the fit was the voice, the track wandered 0.16 Hz off the true
        frequency, and subtracting at the wrong phase added more than it
        removed.
        """
        from scipy import signal

        # Surveyed over the whole file, not only in the pauses. The pauses
        # were tried: hum is there whether anyone speaks or not, and on a low
        # male voice buzz harmonics above the fundamental sat only 0.5 to
        # 1.5 dB over the speech's long-term spectrum. But the quietest third
        # of a 12 s excerpt is four seconds, two Welch segments, and random
        # peaks cleared six decibels: six "harmonics" were subtracted from a
        # recording with no hum at all (-62.8 dB of change on `clean`). The
        # whole file misses buried buzz harmonics and touches nothing that
        # is not there, which is the right way round to be wrong.
        quiet = x
        segment = min(len(quiet), int(2 * rate))
        f, p = signal.welch(quiet, rate, nperseg=segment)
        resolution = f[1] - f[0]
        level = 10 * np.log10(p + 1e-30)
        found = []
        top = min(TOP_HZ, rate / 2 * 0.9)
        k = 1
        while k * nominal < top:
            centre = k * nominal
            # The peak may sit anywhere the drift can take the k-th harmonic.
            reach = max(2 * resolution, k * 0.3)
            near = np.abs(f - centre) <= reach
            ring = (np.abs(f - centre) > reach + resolution) & (
                np.abs(f - centre) <= reach + 12.0)
            if near.any() and ring.sum() >= 4:
                if level[near].max() - np.median(level[ring]) >= PROMINENCE_DB:
                    found.append(k)
            k += 1
        return found

    # ---------------------------------------------------------------- track

    def _score(self, x: np.ndarray, rate: int, f: float,
               harmonics: list[int]) -> float:
        t = np.arange(len(x)) / rate
        return sum(abs(np.mean(x * np.exp(-2j * np.pi * k * f * t))) ** 2
                   for k in harmonics)

    def _track(self, x: np.ndarray, rate: int, nominal: float,
               harmonics: list[int]) -> np.ndarray:
        """The mains frequency per sample, estimated block by block.

        Only on harmonics the survey found, and only on the lowest few of
        them: a higher harmonic moves further for the same drift, which
        sharpens the estimate, but it is also where speech crowds in.
        """
        use = [k for k in harmonics if k * nominal < 1000.0][:6] or harmonics[:1]
        span = int(TRACK_S * rate)
        # Decimated for the search, after a lowpass: aliasing would fold the
        # whole voice onto the harmonics being fitted.
        from scipy import signal

        step = max(1, rate // 2400)
        small = signal.decimate(x, step, ftype="fir") if step > 1 else x
        small_rate = rate / step
        span_small = max(1, span // step)
        grid = np.arange(nominal - SEARCH_HZ, nominal + SEARCH_HZ + 1e-9, 0.02)
        centres, estimates = [], []
        for start in range(0, len(small), span_small):
            piece = small[start : start + span_small]
            if len(piece) < span_small // 2 and estimates:
                break
            scores = [self._score(piece, small_rate, f, use) for f in grid]
            best = grid[int(np.argmax(scores))]
            fine = np.arange(best - 0.02, best + 0.02 + 1e-9, 0.002)
            scores = [self._score(piece, small_rate, f, use) for f in fine]
            estimates.append(fine[int(np.argmax(scores))])
            centres.append((start + len(piece) / 2) * step)
        if len(estimates) > 2:
            # A median over three blocks: one bad block must not drag the
            # track off the mains.
            padded = np.pad(estimates, 1, mode="edge")
            estimates = [float(np.median(padded[i : i + 3]))
                         for i in range(len(estimates))]
        if len(estimates) == 1:
            return np.full(len(x), estimates[0])
        return np.interp(np.arange(len(x)), centres, estimates)

    # ------------------------------------------------------------- process

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        n = len(x)
        self.found = {}
        if n < rate // 2 or not np.any(x):
            return x.astype(np.float32)

        # The analysis copy loses everything below 20 Hz. Infrasonic rumble
        # is common in real recordings — the studio corpus used to test this
        # holds -55 dB of it at 1 Hz — and shifted down with the fundamental
        # it lands at the edge of the decimated band, where the smoothing
        # filter turns it into a transient lasting a second at each end of
        # the file: measured as an estimate six times larger than the hum in
        # the first half-second. At 20 Hz a fourth-order filter costs the
        # 50 Hz fundamental 0.003 dB.
        from scipy import signal

        sos = signal.butter(4, 20.0 / (rate / 2), btype="high", output="sos")
        centred = signal.sosfiltfilt(sos, x - x.mean())
        if self.nominal:
            nominal = self.nominal
            harmonics = self._survey(centred, rate, nominal)
        else:
            surveys = {f: self._survey(centred, rate, f) for f in (50.0, 60.0)}
            nominal = max(surveys, key=lambda f: len(surveys[f]))
            harmonics = surveys[nominal]
        self.found = {"nominal": nominal, "harmonics": []}
        if not harmonics:
            # Nothing stands out: a recording without hum is returned as it
            # came, not with a thousand tiny subtractions of the voice.
            return x.astype(np.float32)

        frequency = self._track(centred, rate, nominal, harmonics)
        phase = 2 * np.pi * np.cumsum(frequency) / rate

        block = max(1, int(BLOCK_S * rate))
        block_rate = rate / block
        positions = np.arange((n + block - 1) // block) * block + block / 2
        # Neighbours for the second, local prominence test, in Hz either
        # side of a harmonic: beyond the notch, inside the same critical band.
        offsets = (-4.0, -2.5, 2.5, 4.0)
        # The neighbours are rotated on the decimated baseband rather than at
        # the full rate: four offsets of a few hertz sit far inside the
        # 100 Hz block rate, and doing it here made the engine four times
        # cheaper per harmonic, where it had run at 3x realtime on buzz.
        t_blocks = positions / rate
        sides = [np.exp(-2j * np.pi * offset * t_blocks) for offset in offsets]

        estimate = np.zeros(n)
        treated = []
        for k in harmonics:
            rotation = np.exp(-1j * k * phase)
            shifted = centred * rotation
            blocks = _blocks(shifted, block)
            steady = _smooth(blocks, self.bandwidth_hz, block_rate)
            level = np.median(np.abs(steady))
            around = [
                np.median(np.abs(_smooth(blocks * side, self.bandwidth_hz, block_rate)))
                for side in sides
            ]
            floor = float(np.median(around)) + 1e-12
            # The survey saw a peak; this checks the tracked harmonic itself
            # stands above its neighbours once drift is followed.
            if 20 * np.log10(level / floor + 1e-12) < LOCAL_PROMINENCE_DB:
                continue
            # Back to per-sample complex amplitude, then back up to the
            # harmonic's own frequency. The factor 2 is the other half of a
            # real sinusoid's spectrum.
            re = np.interp(np.arange(n), positions, steady.real)
            im = np.interp(np.arange(n), positions, steady.imag)
            estimate += 2 * np.real((re + 1j * im) * np.conj(rotation))
            treated.append(k)

        self.found.update(harmonics=treated, mean_frequency=float(frequency.mean()))
        return (x - estimate).astype(np.float32)


@register("dehum")
def _load(argument: str) -> Loaded:
    """``dehum``, ``dehum:50``, ``dehum:60`` or ``dehum:50@2`` (bandwidth)."""
    nominal_text, _, bandwidth_text = argument.partition("@")
    try:
        nominal = float(nominal_text) if nominal_text else None
        bandwidth = float(bandwidth_text) if bandwidth_text else DEFAULT_BANDWIDTH_HZ
    except ValueError:
        raise EngineError(f"dehum takes a frequency and a bandwidth, got {argument!r}")
    if nominal is not None and not 40.0 <= nominal <= 70.0:
        raise EngineError(f"dehum: {nominal:g} Hz is not a mains frequency")
    return Loaded(DehumEngine(nominal, bandwidth),
                  notes={"nominal": nominal, "bandwidth_hz": bandwidth})
