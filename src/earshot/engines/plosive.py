"""Plosive pops, and the blunt instrument they are usually treated with.

    earshot bench --engine deplosive          # dynamic low band below 150 Hz
    earshot bench --engine deplosive:200      # a higher corner
    earshot bench --engine highpass:80        # the global filter, for contrast

A pop is breath hitting the capsule: tens of milliseconds of energy below
150 Hz at the start of a word. The usual fix is a high-pass on the whole
track, and it has a cost that is easy to miss on a light voice and obvious
on a low one — the fundamental of a low male voice sits at 80 to 120 Hz, and
a filter that removes the pop removes the chest of the voice with it.

``deplosive`` acts only where a pop is. It learns, per file, how much low
band this speaker's voice carries relative to the speech band, and wherever
the low band rises far above that prediction it pulls the low band down by
the excess, 5 ms at a time, and lets it back up over 60 ms. Where the voice
behaves as the voice does, the output is the input. ``highpass`` is here as the thing to beat: if the local filter does
not measurably beat the global one on `plosive` while leaving `clean` alone,
it is not worth its complexity.

The detector's thresholds are guesses. No pops have been measured in this
project's own material yet, and the `plosive` recipe they are tested against
is modelled rather than fitted.
"""

from __future__ import annotations

import numpy as np

from . import EngineError, Loaded, register

# Analysis frame for the band levels.
FRAME_S = 0.005

# The band that is limited. A pop's energy sits below this; so does the
# fundamental of a low male voice, which is why it is limited only where it
# exceeds what the voice itself predicts.
DEFAULT_CORNER_HZ = 150.0

# The band the prediction is made from: where the voice lives and a pop
# does not reach.
REFERENCE_HZ = (300.0, 3000.0)

# How far above the speaker's own usual low-to-mid balance the low band may
# rise before it is pulled down. A first version detected events with a
# fixed low-over-mid ratio of 6 dB and fired seven to nine times in twelve
# seconds of clean male speech while missing half the pops: nasals and
# vowels of a low voice cross any fixed ratio. The balance is per speaker,
# so it is measured per file.
#
# 18 dB from a sweep on three EARS speakers, two 12 s excerpts each, with
# the `plosive` recipe: 12, 15, 18 and 21 dB all took the pops in the
# 30-200 Hz band down by 8 to 18 dB, while the change to the same band on
# *clean* male speech fell from -55..-59 dB at 12 to -58..-62 dB at 18. The
# clean male excerpts still trigger 12 to 23 short cuts; whether those are
# real pops in the studio corpus or the voice is a listening question.
MARGIN_DB = 18.0

# Never pull the low band down further than this.
MAX_CUT_DB = 24.0

# The gain moves no faster than this, so the cut does not modulate a vowel.
ATTACK_S, RELEASE_S = 0.005, 0.06


def _split(x: np.ndarray, rate: int, corner: float):
    """Low band and everything else, summing back to the input exactly."""
    from scipy import signal

    sos = signal.butter(4, corner / (rate / 2), btype="low", output="sos")
    low = signal.sosfiltfilt(sos, x)
    return low, x - low


def _highpass(x: np.ndarray, rate: int, corner: float, order: int = 4) -> np.ndarray:
    from scipy import signal

    sos = signal.butter(order, corner / (rate / 2), btype="high", output="sos")
    # Zero-phase, so the output lines up with the input sample for sample.
    return signal.sosfiltfilt(sos, x)


def _levels(band: np.ndarray, frame: int) -> np.ndarray:
    count = len(band) // frame
    power = np.mean(band[: count * frame].reshape(count, frame) ** 2, axis=1)
    return 10 * np.log10(power + 1e-20)


class DeplosiveEngine:
    """A dynamic low band: pulled down only where it outruns the voice."""

    def __init__(self, corner: float = DEFAULT_CORNER_HZ):
        self.corner = corner
        self.name = ("deplosive" if corner == DEFAULT_CORNER_HZ
                     else f"deplosive({corner:g}Hz)")
        self.events: list[tuple[float, float]] = []

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from scipy import signal

        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        self.events = []
        frame = max(1, int(FRAME_S * rate))
        if len(x) < frame * 16 or not np.any(x):
            return x.astype(np.float32)

        # Measured on a reflected copy, so neither band starts with a filter
        # transient: without it the first and last 50-175 ms of any excerpt
        # — and so of every chunk of a long file — read as a pop.
        # Odd extension, as filtfilt itself uses: a plain mirror puts a kink
        # at the join, and a kink has low-frequency content of its own.
        edge = min(len(x) - 1, int(0.1 * rate))
        padded = np.concatenate([2 * x[0] - x[edge:0:-1], x,
                                 2 * x[-1] - x[-2 : -edge - 2 : -1]])

        def band(lo: float, hi: float) -> np.ndarray:
            sos = signal.butter(4, [lo / (rate / 2), hi / (rate / 2)],
                                btype="band", output="sos")
            return _levels(signal.sosfiltfilt(sos, padded)[edge : edge + len(x)], frame)

        # The low band from 30 Hz for measuring: real recordings carry
        # infrasonic rumble (-55 dB at 1 Hz in the corpus used here), which
        # is not a pop and must not count as one.
        low_level = band(30.0, self.corner)
        reference = band(*REFERENCE_HZ)

        speech = reference > reference.max() - 40.0
        if speech.sum() < 10:
            return x.astype(np.float32)
        balance = float(np.median((low_level - reference)[speech]))
        expected = reference + balance + MARGIN_DB
        # In a pause the reference collapses and every trace of low-band
        # noise would read as excess; a pop there must also stand clear of
        # the file's own low-band floor.
        floor = float(np.percentile(low_level, 20)) + MARGIN_DB
        # And it must be loud in absolute terms, within 30 dB of the loudest
        # speech: a signal with next to nothing below 150 Hz otherwise shows
        # "excess" over nothing at every edge and every kink.
        audible = low_level > reference.max() - 30.0
        excess = np.where(audible, low_level - np.maximum(expected, floor), 0.0)
        cut = np.clip(excess, 0.0, MAX_CUT_DB)

        # Attack fast, release slowly, in the frame domain: a running maximum
        # over the release, then a short average for the attack.
        release = max(1, int(RELEASE_S / FRAME_S))
        held = np.array([cut[max(0, i - release): i + 1].max() for i in range(len(cut))])
        attack = max(1, int(ATTACK_S / FRAME_S) * 2 + 1)
        held = np.convolve(held, np.ones(attack) / attack, mode="same")

        hot = held > 3.0
        edges = np.diff(np.concatenate(([0], hot.astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            self.events.append((start * frame / rate, stop * frame / rate))
        if not self.events:
            return x.astype(np.float32)

        centres = (np.arange(len(held)) + 0.5) * frame
        gain = 10 ** (-np.interp(np.arange(len(x)), centres, held) / 20)
        low, rest = _split(x, rate, self.corner)
        return (rest + low * gain).astype(np.float32)


class HighpassEngine:
    """A plain high-pass on everything. The baseline ``deplosive`` must beat."""

    def __init__(self, corner: float = 80.0):
        self.corner = corner
        self.name = f"highpass({corner:g}Hz)"

    def process(self, audio: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        if len(x) < 64:
            return x.astype(np.float32)
        return _highpass(x, rate, self.corner).astype(np.float32)


def _corner(argument: str, default: float) -> float:
    try:
        corner = float(argument) if argument else default
    except ValueError:
        raise EngineError(f"expected a corner frequency in Hz, got {argument!r}")
    if not 20.0 <= corner <= 400.0:
        raise EngineError(f"{corner:g} Hz is not a plausible high-pass corner")
    return corner


@register("deplosive")
def _load_deplosive(argument: str) -> Loaded:
    corner = _corner(argument, DEFAULT_CORNER_HZ)
    return Loaded(DeplosiveEngine(corner), notes={"corner_hz": corner})


@register("highpass")
def _load_highpass(argument: str) -> Loaded:
    corner = _corner(argument, 80.0)
    return Loaded(HighpassEngine(corner), notes={"corner_hz": corner})
