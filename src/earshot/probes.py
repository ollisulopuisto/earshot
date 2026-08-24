"""The questions the bench asks, and how each one is answered.

A probe is one question with one number. Together they are meant to be
enough to choose an engine without listening — not because listening is
unnecessary, but because listening cannot be repeated identically, cannot
be run in CI, and cannot tell you that the 14 kHz band is invented.

The set is deliberately small. Each probe here exists because it changed a
decision at least once:

``recovery``      Did it get the missing band back? The only probe that
                  needs a clean reference, and the only one that can say an
                  engine restored rather than merely altered.
``origin``        Is the output still the input, band by band? Separates a
                  mask from a synthesiser. Nothing else reveals this.
``cleanup``       Did the floor come down without the speech coming with it?
``preservation``  What happens to sound that is not a voice? A speech-only
                  model deletes it, which is fine for a lone microphone and
                  ruinous for a recording with a room in it.
``stability``     Same input twice, how close? Decides whether results can
                  be cached and whether a measured difference is the setting
                  or the dice. Reported as a margin, not a yes/no — the
                  first plug-in measured was neither bit-exact nor unstable.
``throughput``    How much faster than realtime, and on how many cores.

Every probe returns ``Result`` objects with ``better`` set, so the report
can rank without knowing what any of it means.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass, field

import numpy as np

from . import degrade, metrics
from .engines import Loaded, check_contract


@dataclass
class Result:
    """One measurement, with enough context to be read a year later."""

    probe: str
    metric: str
    value: float
    unit: str = "dB"
    better: str = "lower"  # "lower", "higher" or "" when it is not a score
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.probe}/{self.metric} = {self.value:.2f} {self.unit}".strip()


@dataclass
class Run:
    """Everything one engine produced on one piece of material."""

    engine: str
    damage: str
    results: list[Result] = field(default_factory=list)
    skipped: str = ""
    failed: str = ""

    def value(self, probe: str, metric: str) -> float | None:
        for result in self.results:
            if result.probe == probe and result.metric == metric:
                return result.value
        return None


# Bands the probes report in. Chosen to match how speech restoration fails:
# the bottom is rumble and proximity, 1–4 kHz is intelligibility and where
# engines do most of their work, above 12 kHz is air — the part most often
# invented rather than recovered.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", 100.0, 1000.0),
    ("mid", 1000.0, 4000.0),
    ("high", 6000.0, 12000.0),
    ("air", 14000.0, 20000.0),
)


def run_all(
    loaded: Loaded,
    clean: np.ndarray,
    rate: int,
    damage: degrade.Damage,
    reference_band: tuple[float, float] | None = None,
) -> Run:
    """Damage the clean signal, restore it, and answer every question.

    The clean signal is the reference throughout. Probes that do not need
    it still use the *damaged* signal as their input, because that is what
    the engine was given and a "before" that the engine never saw would
    make the numbers unreadable.
    """
    run = Run(engine=loaded.engine.name, damage=damage.name)
    if damage.needs and not _have(damage.needs):
        run.skipped = f"needs {damage.needs}"
        return run

    try:
        broken = np.asarray(damage.apply(np.asarray(clean, dtype=np.float32), rate))
    except Exception as exc:
        run.skipped = f"could not apply damage: {exc}"
        return run

    try:
        started = time.perf_counter()
        cpu_before = _cpu_seconds()
        restored = check_contract(
            broken, loaded.engine.process(broken, rate), loaded.engine.name
        )
        wall = time.perf_counter() - started
        cpu = _cpu_seconds() - cpu_before
    except Exception as exc:
        run.failed = str(exc)
        return run

    clean = np.asarray(clean, dtype=np.float32)[: len(restored)]

    # --- recovery: how close to the original, in the band the damage hit
    low, high = reference_band or _damaged_band(damage, rate)
    if high > low:
        before = metrics.log_spectral_distance(broken, clean, rate, low, high)
        after = metrics.log_spectral_distance(restored, clean, rate, low, high)
        run.results += [
            Result("recovery", "lsd_before", before, "dB", "lower",
                   f"{low:.0f}–{high:.0f} Hz, damaged vs clean"),
            Result("recovery", "lsd_after", after, "dB", "lower",
                   f"{low:.0f}–{high:.0f} Hz, restored vs clean"),
            Result("recovery", "gained", before - after, "dB", "higher",
                   "positive means closer to the original than the damage was"),
        ]

    # --- origin: filtering or invention, band by band
    for name, band_low, band_high in BANDS:
        if band_low >= rate / 2:
            continue
        run.results.append(
            Result(
                "origin",
                name,
                metrics.band_correlation(broken, restored, rate, band_low, band_high),
                "r",
                "",
                f"{band_low:.0f}–{band_high:.0f} Hz; ~1 filtered, ~0 synthesised",
            )
        )

    # --- cleanup: floor down, speech left alone
    before_dynamics = metrics.dynamics(broken, rate)
    after_dynamics = metrics.dynamics(restored, rate)
    run.results += [
        Result("cleanup", "floor_change", after_dynamics.floor - before_dynamics.floor,
               "dB", "lower", "negative is a quieter noise floor"),
        Result("cleanup", "speech_change", after_dynamics.speech - before_dynamics.speech,
               "dB", "", "should be near zero: cleaning is not turning down"),
        Result("cleanup", "range_gained",
               after_dynamics.range_db - before_dynamics.range_db, "dB", "higher",
               "speech-to-silence distance opened up by this much"),
    ]

    # --- stability
    repeat = check_contract(
        broken, loaded.engine.process(broken, rate), loaded.engine.name
    )
    run.results.append(
        Result("stability", "overall", metrics.repeatability(restored, repeat),
               "dB", "higher",
               "how far below the signal two runs of the same input differ; "
               ">120 is bit-identical, >60 is inaudible but not a pure function")
    )
    # Per band, because the aggregate hides the thing worth knowing. Measured
    # on a commercial plug-in: below 1 kHz it repeats to +74 dB, and in the
    # band it invents the two runs differ by *more than the signal* (-1 dB).
    # An engine can be a pure function where it filters and a dice roll where
    # it generates, and only a per-band reading says so.
    from scipy import signal as _sig

    for name, band_low, band_high in BANDS:
        if band_low >= rate / 2:
            continue
        sos = _sig.butter(
            4, [band_low / (rate / 2), min(band_high, rate / 2 * 0.99) / (rate / 2)],
            btype="band", output="sos",
        )
        run.results.append(
            Result("stability", name,
                   metrics.repeatability(_sig.sosfilt(sos, restored),
                                         _sig.sosfilt(sos, repeat)),
                   "dB", "higher",
                   f"{band_low:.0f}–{band_high:.0f} Hz; low means this band is "
                   "generated afresh each run")
        )

    # --- throughput
    seconds_of_audio = len(broken) / rate
    run.results += [
        Result("throughput", "realtime", seconds_of_audio / max(wall, 1e-6), "x",
               "higher", "seconds of audio per second of wall clock"),
        Result("throughput", "cores", cpu / max(wall, 1e-6), "", "",
               "CPU seconds per wall second; 1.0 means single-threaded"),
        Result("throughput", "load", loaded.load_seconds, "s", "lower",
               "one-off cost of getting the engine ready"),
    ]
    return run


def preservation(loaded: Loaded, rate: int = 48000, seconds: float = 5.0) -> Run:
    """What the engine does to sound that is not a voice.

    Its own probe because it needs its own material: a speech signal cannot
    answer it. A logarithmic sweep covers every band with the same energy,
    so the answer comes out per octave rather than as one number.

    A speech-only model scores tens of decibels here. That is not a bug in
    the model — it is the correct behaviour for a lone microphone, and it is
    exactly wrong for a microphone that also has a room, a guitar, or a
    second person's laugh in it. The bench states the number and leaves the
    judgement to whoever knows the material.
    """
    from scipy import signal as _signal

    run = Run(engine=loaded.engine.name, damage="sweep")
    t = np.arange(int(seconds * rate)) / rate
    sweep = (0.2 * _signal.chirp(t, 100, seconds, 16000, method="logarithmic")).astype(
        np.float32
    )
    try:
        out = check_contract(sweep, loaded.engine.process(sweep, rate), loaded.engine.name)
    except Exception as exc:
        run.failed = str(exc)
        return run

    run.results.append(
        Result("preservation", "overall", metrics.suppression(sweep, out), "dB", "lower",
               "how much of a non-speech signal disappeared")
    )
    # The sweep is logarithmic, so the time at which it passes a frequency is
    # known and each octave can be looked at where it actually happened.
    span = np.log(16000 / 100)
    for low, high in ((100, 500), (500, 2000), (2000, 6000), (6000, 16000)):
        window = (t >= seconds * np.log(low / 100) / span) & (
            t < seconds * np.log(high / 100) / span
        )
        if window.sum() > rate // 100:
            run.results.append(
                Result("preservation", f"{low}-{high}Hz",
                       metrics.suppression(sweep[window], out[window]), "dB", "lower",
                       "positive means this octave was removed")
            )
    return run


def _damaged_band(damage: degrade.Damage, rate: int) -> tuple[float, float]:
    """Which band to score recovery in, given what the damage did.

    Band-limiting removes a specific region and that is where the question
    lies. Everything else damages the whole signal, so the whole signal is
    the band.
    """
    ceiling = min(16000.0, rate / 2 * 0.95)
    for function, options in damage.steps:
        if function is degrade.band_limit:
            high = float(options.get("high") or 0)
            if 0 < high < ceiling:
                return high, ceiling
    return 100.0, ceiling


def _have(tool: str) -> bool:
    import shutil

    return shutil.which(tool) is not None


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def default_material(rate: int = 48000, seconds: float = 12.0) -> np.ndarray:
    """Synthetic speech-like material, for when there is no recording.

    Not a substitute for real voices and not offered as one: it exists so
    the bench, its tests and CI can run anywhere. It is a pulse train
    through moving formant resonances with pauses — enough structure that
    the dynamics and band metrics mean something, and enough silence that
    the noise floor is measurable.
    """
    from scipy import signal as _signal

    rng = np.random.default_rng(11)
    n = int(seconds * rate)
    t = np.arange(n) / rate

    f0 = 110 + 20 * np.sin(2 * np.pi * 0.7 * t)
    excitation = _signal.square(2 * np.pi * np.cumsum(f0) / rate, duty=0.06)
    excitation += 0.05 * rng.normal(0, 1, n)

    out = np.zeros(n)
    for centre, base, gain in ((520, 620, 1.0), (1200, 900, 0.5), (2600, 1200, 0.25)):
        sweep = centre + base * 0.25 * np.sin(2 * np.pi * 1.3 * t + centre)
        sos = _signal.butter(
            2, [np.clip(sweep.mean() * 0.6, 60, rate / 2 - 100) / (rate / 2),
                np.clip(sweep.mean() * 1.6, 120, rate / 2 - 50) / (rate / 2)],
            btype="band", output="sos",
        )
        out += gain * _signal.sosfilt(sos, excitation)

    # Pauses, so there is a floor to measure between the "words".
    gate = np.ones(n)
    for start in range(0, n, int(1.5 * rate)):
        gate[start : start + int(0.5 * rate)] = 0.0
    ramp = int(0.02 * rate)
    gate = np.convolve(gate, np.ones(ramp) / ramp, mode="same")

    out *= gate
    out /= np.abs(out).max() + 1e-9
    out *= 0.3
    out += 10 ** (-60 / 20) * rng.normal(0, 1, n)  # a real recording has a floor
    return out.astype(np.float32)
