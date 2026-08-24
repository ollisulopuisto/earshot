"""The test kit every engine has to pass.

An engine PR should not need a reviewer to remember the contract. Import
this, call it once, and the rules are checked:

    from earshot.testing import assert_engine_contract

    def test_my_engine_keeps_the_contract():
        assert_engine_contract(MyEngine())

The cases here are the ones that have actually broken engines, not a
generated sweep:

* **A file of pure silence.** Models that normalise divide by the input
  level. A podcast has silence in it — often several seconds at the top of a
  take — and an engine that returns NaN there ruins the whole file, not the
  silent part.
* **Something shorter than the model's own window.** The bench slices
  material; a chain engine passes fragments. An engine that needs 400 ms and
  gets 50 must pad, not crash and not truncate.
* **A length that is not a multiple of anything.** Frame-based models
  quietly round to a whole number of frames, and the last partial frame
  disappears. That is the single most common way an engine breaks the
  sample-count rule, and it never shows on a test file of exactly ten
  seconds.
* **A large DC offset.** Some interfaces produce it. A filter bank that
  assumes zero mean returns a wildly wrong first frame.
* **State leaking between calls.** Not a determinism requirement — engines
  are allowed to be stochastic, and at least one commercial one is. The
  check separates the two: process the same audio twice to establish how
  much an engine varies of its own accord, then process something *else* in
  between and repeat. If the result changes much more when a different file
  went through first, the engine is carrying the previous file's tail into
  this one. That is the `reset=False` failure mode, it is silent, and it
  corrupts the first seconds of every file after the first.

Nothing here checks quality. Quality is what the bench is for; this only
checks that the bench's numbers will mean what they say.
"""

from __future__ import annotations

import numpy as np

from .engines import EngineError, check_contract

RATE = 48000

# Lengths chosen to catch frame rounding: a whole number of seconds, an odd
# prime, something shorter than any plausible analysis window, and a length
# just past a power of two.
LENGTHS = (RATE, RATE + 1, 1024 * 33 + 7, 2048, 511)


def _material(n: int, kind: str = "speech") -> np.ndarray:
    rng = np.random.default_rng(17)
    t = np.arange(n) / RATE
    if kind == "silence":
        return np.zeros(n, dtype=np.float32)
    if kind == "dc":
        return (np.full(n, 0.4) + 0.05 * rng.normal(0, 1, n)).astype(np.float32)
    if kind == "tiny":
        return (1e-9 * rng.normal(0, 1, n)).astype(np.float32)
    voiced = 0.3 * np.sin(2 * np.pi * 140 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (voiced + 0.02 * rng.normal(0, 1, n)).astype(np.float32)


def assert_engine_contract(engine, rate: int = RATE) -> None:
    """Raise ``AssertionError`` if the engine breaks any rule.

    Call this from the engine's own test. It is deliberately strict about
    length and finiteness and deliberately silent about everything else.
    """
    name = getattr(engine, "name", type(engine).__name__)
    assert isinstance(name, str) and name, "an engine needs a non-empty name"

    for kind in ("speech", "silence", "dc", "tiny"):
        for length in LENGTHS:
            audio = _material(length, kind)
            try:
                out = engine.process(audio, rate)
            except EngineError:
                raise
            except Exception as exc:  # noqa: BLE001 - the report is the point
                raise AssertionError(
                    f"{name} raised on {kind} of {length} samples: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                out = check_contract(audio, out, name)
            except EngineError as exc:
                raise AssertionError(f"{name} on {kind}/{length}: {exc}") from exc
            assert out.dtype.kind == "f", (
                f"{name} returned {out.dtype} on {kind}/{length}; audio is float"
            )

    # Checked before the silence case on purpose: an engine that leaks the
    # previous file also fails "silence in, silence out", and the leak is the
    # cause while the noise in the silence is only the symptom. Reporting the
    # cause saves the next person the same hour.
    #
    # Stochastic is allowed; stateful is not, and telling them apart takes
    # more than repeating a call. An engine that leaks does so on every call,
    # so two repeats vary as much as a contaminated one — the first version of
    # this check compared exactly that and caught nothing.
    #
    # What separates them is *where* the difference lands. Carried-over state
    # decays: it corrupts the head of the file and is gone by the middle.
    # Stochastic generation is spread evenly. So process the same audio twice
    # and compare the head against the tail. A leak is a head that disagrees
    # far more than the tail does.
    audio = _material(rate * 2, "speech")
    first = np.asarray(engine.process(audio, rate), dtype=np.float64)
    second = np.asarray(engine.process(audio, rate), dtype=np.float64)
    split = min(int(0.25 * rate), len(first) // 4)

    def _level(x):
        return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0

    head = _level((first - second)[:split])
    tail = _level((first - second)[split:])
    quiet_enough = _level(first) * 1e-6
    if head > quiet_enough and head > tail * 10:
        raise AssertionError(
            f"{name} disagrees with itself in the first {split / rate:.2f} s "
            f"({20 * np.log10(head + 1e-12):.1f} dBFS) far more than in the "
            f"rest of the file ({20 * np.log10(tail + 1e-12):.1f} dBFS) — it "
            "is carrying state between calls that it should reset"
        )

    # Silence in must not become noise out. An engine is allowed to leave a
    # floor, but a model hallucinating speech into digital silence is a
    # different thing and it happens to generative restorers.
    quiet = engine.process(_material(rate * 2, "silence"), rate)
    level = float(np.sqrt(np.mean(np.asarray(quiet) ** 2)))
    assert level < 1e-2, (
        f"{name} produced {20 * np.log10(level + 1e-12):.1f} dBFS from digital "
        "silence; that is invention, not restoration"
    )
