"""Tests for responsive CPU/GPU concurrency and thread budgeting."""

import sys
import types

import numpy as np
import soundfile as sf

from earshot import cli, engines

RATE = 48000


def test_cpu_thread_budget_leaves_cores_free(monkeypatch):
    """Ensure cpu_thread_budget leaves reserve cores free for desktop responsiveness."""
    monkeypatch.delenv("EARSHOT_NUM_THREADS", raising=False)
    monkeypatch.delenv("EARSHOT_WORKERS", raising=False)

    monkeypatch.setattr("os.cpu_count", lambda: 8)
    if hasattr(sys.modules["os"], "sched_getaffinity"):
        monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(8)))
    if hasattr(sys.modules["os"], "process_cpu_count"):
        monkeypatch.setattr("os.process_cpu_count", lambda: 8)

    assert engines.cpu_thread_budget(reserve=1) == 7
    assert engines.cpu_thread_budget(reserve=2) == 6


def test_cpu_thread_budget_floor_of_one(monkeypatch):
    """On single-core or heavily restricted systems, it never falls below 1."""
    monkeypatch.delenv("EARSHOT_NUM_THREADS", raising=False)
    monkeypatch.delenv("EARSHOT_WORKERS", raising=False)

    monkeypatch.setattr("os.cpu_count", lambda: 1)
    if hasattr(sys.modules["os"], "sched_getaffinity"):
        monkeypatch.setattr("os.sched_getaffinity", lambda _pid: {0})
    if hasattr(sys.modules["os"], "process_cpu_count"):
        monkeypatch.setattr("os.process_cpu_count", lambda: 1)

    assert engines.cpu_thread_budget(reserve=1) == 1


def test_cpu_thread_budget_env_override(monkeypatch):
    """Environment variables override automatic CPU detection."""
    monkeypatch.setenv("EARSHOT_NUM_THREADS", "4")
    assert engines.cpu_thread_budget() == 4

    monkeypatch.delenv("EARSHOT_NUM_THREADS")
    monkeypatch.setenv("EARSHOT_WORKERS", "3")
    assert engines.cpu_thread_budget() == 3


def test_torch_device_multiple_cuda(monkeypatch):
    """torch_device distributes workers across cuda:0, cuda:1 when multiple exist."""
    monkeypatch.delenv("EARSHOT_DEVICE", raising=False)
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert engines.torch_device(0) == "cuda:0"
    assert engines.torch_device(1) == "cuda:1"
    assert engines.torch_device(2) == "cuda:0"


def test_configure_torch_threads(monkeypatch):
    """configure_torch_threads calls torch.set_num_threads."""
    called_with = []
    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda n: called_with.append(n)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    engines.configure_torch_threads(4)
    assert called_with == [4]


def test_restore_parallel_workers(tmp_path, monkeypatch, capsys):
    """earshot restore parallelizes multi-file processing across workers."""
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir()

    for i in range(3):
        x = np.full(RATE, 0.1 * (i + 1), dtype=np.float32)
        sf.write(in_dir / f"take_{i}.wav", x, RATE)

    rc = cli.main([
        "restore", str(in_dir), "-o", str(out_dir),
        "--engine", "passthrough", "--workers", "2",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("*.wav"))
    assert len(out_files) == 3


def test_bench_parallel_jobs(monkeypatch, capsys):
    """earshot bench evaluates excerpts/recipes in parallel with --jobs."""
    rc = cli.main([
        "bench", "--damage", "clean", "--damage", "hiss",
        "--engine", "passthrough", "--seconds", "1.0",
        "--jobs", "2",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# earshot bench" in out
    assert "clean" in out
    assert "hiss" in out
