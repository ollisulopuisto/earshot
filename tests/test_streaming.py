"""Streaming restore: the file is no longer held in memory, and the audio
must not notice.

``engines.process_file_in_chunks`` is ``process_in_chunks`` with streamed
I/O. The output has to be the same samples - an editor stacking the result
against the original hears every difference - and peak memory has to stop
scaling with the length of the file. These tests hold both halves.
"""

import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

from earshot import engines, metrics

RATE = 48000


class Gain:
    name = "gain"

    def process(self, audio, rate):
        return np.asarray(audio, dtype=np.float32) * 0.5


class Lowpass:
    """An engine with memory; a stateless one would pass a broken scheme."""

    name = "lowpass"

    def process(self, audio, rate):
        from scipy import signal
        sos = signal.butter(4, 3000.0, "low", fs=rate, output="sos")
        return signal.sosfilt(sos, np.asarray(audio, dtype=np.float64)).astype(np.float32)


@pytest.mark.parametrize("engine", [Gain(), Lowpass()], ids=["gain", "lowpass"])
@pytest.mark.parametrize("seconds", [0.5, 2.0, 2.5, 4.25, 7.0, 9.13])
def test_streamed_output_matches_the_array_chunker(tmp_path, engine, seconds):
    """Same chunking, same fades, same single-pass path - same samples."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(int(seconds * RATE)).astype(np.float32) * 0.1
    source = tmp_path / "in.wav"
    target = tmp_path / "out.wav"
    sf.write(source, x, RATE)

    frames, rate = engines.process_file_in_chunks(
        engine, source, target, chunk_seconds=2.0, overlap_seconds=0.25)
    assert (frames, rate) == (len(x), RATE)

    # What the stream actually processed: the file's samples, not the
    # float32 array they were written from (the file is 16-bit by default).
    x_on_disk, _ = sf.read(source, dtype="float32")
    expected = engines.process_in_chunks(engine, x_on_disk, RATE,
                                         chunk_seconds=2.0,
                                         overlap_seconds=0.25)
    got, got_rate = sf.read(target, dtype="float32")
    assert got_rate == RATE and len(got) == len(expected)
    # The file is PCM_24; the array is float32. Quantisation is the only
    # allowed difference.
    np.testing.assert_allclose(got, expected, atol=2**-22)


def test_on_block_tiles_the_whole_recording(tmp_path):
    """The callback's regions are contiguous and cover every sample, or a
    meter built on them silently drops or double-counts audio."""
    seconds = 5.37
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int(seconds * RATE)).astype(np.float32) * 0.1
    source = tmp_path / "in.wav"
    sf.write(source, x, RATE)

    seen = []
    engines.process_file_in_chunks(
        Gain(), source, tmp_path / "out.wav",
        chunk_seconds=2.0, overlap_seconds=0.25,
        on_block=lambda start, a, b: seen.append((start, len(a), len(b))))

    assert seen[0][0] == 0
    for (start, na, nb), (next_start, _, _) in zip(seen, seen[1:]):
        assert na == nb and start + na == next_start
    assert seen[-1][0] + seen[-1][1] == len(x)


def test_dynamics_meter_equals_dynamics_over_the_whole():
    rng = np.random.default_rng(11)
    x = (rng.standard_normal(int(9.7 * RATE)) * 0.05).astype(np.float32)
    x[: RATE] *= 40.0  # give the loudest and quietest fifths something to sort
    wanted = metrics.dynamics(x, RATE)

    meter = metrics.DynamicsMeter(RATE)
    pos = 0
    for size in (12345, 1, 400_000, 77_777, 19_200, 999):
        meter.add(x[pos : pos + size])
        pos += size
    meter.add(x[pos:])
    got = meter.result()

    assert got.speech == pytest.approx(wanted.speech, abs=1e-9)
    assert got.floor == pytest.approx(wanted.floor, abs=1e-9)


def test_dynamics_meter_shorter_than_one_frame():
    """dynamics() scores a short signal as a single short frame; so does
    the meter."""
    x = np.full(1000, 0.25, dtype=np.float32)
    meter = metrics.DynamicsMeter(RATE)
    meter.add(x[:400])
    meter.add(x[400:])
    wanted = metrics.dynamics(x, RATE)
    got = meter.result()
    assert got.speech == pytest.approx(wanted.speech, abs=1e-9)
    assert got.floor == pytest.approx(wanted.floor, abs=1e-9)


def test_peak_memory_stops_following_the_file(tmp_path):
    """The point of the change. A four minute file is 74 MB as float32;
    the array path holds it twice plus the chunker's copy, the streamed
    path holds a chunk. Measured in fresh processes because ru_maxrss
    only grows."""
    seconds = 240.0
    source = tmp_path / "long.wav"
    with sf.SoundFile(str(source), "w", samplerate=RATE, channels=1,
                      subtype="PCM_16") as f:
        rng = np.random.default_rng(5)
        for _ in range(24):
            f.write((rng.standard_normal(10 * RATE) * 0.05).astype(np.float32))

    if not pathlib.Path("/proc/self/status").exists():
        pytest.skip("peak-memory accounting needs /proc (Linux)")

    script = """
import sys
import numpy as np
import soundfile as sf
from earshot import engines

class Gain:
    name = "gain"
    def process(self, audio, rate):
        return np.asarray(audio, dtype=np.float32) * 0.5

def vmpeak_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmPeak"):
            return int(line.split()[1])
    raise RuntimeError("no VmPeak")

mode, source, target = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "base":
    pass  # interpreter and imports only: the yardstick for the other two
elif mode == "array":
    x, rate = sf.read(source, dtype="float32")
    y = engines.process_in_chunks(Gain(), x, rate, chunk_seconds=20.0,
                                  overlap_seconds=1.0)
    sf.write(target, y, rate, subtype="PCM_24")
else:
    engines.process_file_in_chunks(Gain(), source, target,
                                   chunk_seconds=20.0, overlap_seconds=1.0)
print(vmpeak_kb())
"""
    # A minimal environment, so what the parent test runner happened to
    # import or set cannot leak into either measurement.
    import os
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
           "EARSHOT_NO_DOWNLOAD": "1"}
    peaks = {}
    for mode in ("base", "array", "stream"):
        out = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script), mode,
             str(source), str(tmp_path / f"{mode}.wav")],
            capture_output=True, text=True, check=True, env=env)
        peaks[mode] = int(out.stdout.strip())
    # VmPeak, not RSS: virtual mappings are per-process, while this
    # development sandbox's RSS accounting charges the child for its
    # parent's pages and makes RSS comparisons meaningless. The delta over
    # the imports-only baseline is what each approach actually allocates.
    file_kb = seconds * RATE * 4 / 1024  # the file as one float32 array
    # Sanity: the measurement can tell - the array path holds the file at
    # least twice over.
    assert peaks["array"] - peaks["base"] > 1.5 * file_kb, peaks
    # The point: streaming adds a few chunks on top of the interpreter,
    # not the episode. A whole-file regression would add at least two
    # full files and fail this by a wide margin.
    assert peaks["stream"] - peaks["base"] < 0.75 * file_kb, peaks


class FailsOnSecondCall:
    """The engine that dies halfway through a real file: the first chunk is
    fine, the damage is already written, and then this."""

    name = "fragile"

    def __init__(self, exc=RuntimeError("model exploded")):
        self.calls = 0
        self.exc = exc

    def process(self, audio, rate):
        self.calls += 1
        if self.calls == 2:
            raise self.exc
        return np.asarray(audio, dtype=np.float32)


def _four_second_source(tmp_path):
    x = np.random.default_rng(9).standard_normal(4 * RATE).astype(np.float32) * 0.1
    source = tmp_path / "in.wav"
    sf.write(source, x, RATE)
    return source


def test_a_late_failure_leaves_no_truncated_target(tmp_path):
    """Two chunks in of four, the engine dies: nothing under the final
    name, and no partial file left behind either."""
    source = _four_second_source(tmp_path)
    target = tmp_path / "out.wav"
    with pytest.raises(RuntimeError, match="model exploded"):
        engines.process_file_in_chunks(
            FailsOnSecondCall(), source, target,
            chunk_seconds=1.0, overlap_seconds=0.1)
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_a_late_failure_preserves_the_existing_target(tmp_path):
    """The array path only wrote after the engine succeeded, so a rerun
    over yesterday's result could not destroy it. Streaming keeps that."""
    source = _four_second_source(tmp_path)
    target = tmp_path / "out.wav"
    sf.write(target, np.full(RATE, 0.5, dtype=np.float32), RATE)
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match="model exploded"):
        engines.process_file_in_chunks(
            FailsOnSecondCall(), source, target,
            chunk_seconds=1.0, overlap_seconds=0.1)
    assert target.read_bytes() == before


def test_an_interrupt_cleans_up_and_preserves_the_target(tmp_path):
    """Ctrl-C is not an Exception; the guard is finally, not except."""
    source = _four_second_source(tmp_path)
    target = tmp_path / "out.wav"
    sf.write(target, np.full(RATE, 0.5, dtype=np.float32), RATE)
    before = target.read_bytes()
    with pytest.raises(KeyboardInterrupt):
        engines.process_file_in_chunks(
            FailsOnSecondCall(exc=KeyboardInterrupt()), source, target,
            chunk_seconds=1.0, overlap_seconds=0.1)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))


def test_the_target_parent_directory_is_created(tmp_path):
    """audio_io.write made the parents; the streaming path must too."""
    source = _four_second_source(tmp_path)
    target = tmp_path / "new" / "nested" / "out.wav"
    engines.process_file_in_chunks(Gain(), source, target,
                                   chunk_seconds=2.0, overlap_seconds=0.25)
    assert sf.info(str(target)).frames == 4 * RATE
