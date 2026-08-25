"""The record has to say which machine produced a number.

``throughput`` is meaningless without the machine, and two Apple Silicon
machines running the same macOS produce the same ``platform.platform()`` and
the same ``platform.processor()``. A scoreboard that labels both ``arm`` is
not recording the machine; it only looks as if it is.
"""

import json
import platform
import subprocess

import pytest

from earshot import report


def _results(tmp_path, cpu: str) -> str:
    """One stored result file, as if measured on ``cpu``.

    The two files differ only in the CPU, exactly as two Apple Silicon Macs
    of different generations do.
    """
    runs = [{"engine": "lavasr", "damage": "clean", "material": "",
             "results": [{"probe": "speed", "metric": "realtime",
                          "value": 77.0, "better": "higher"}]}]
    path = tmp_path / f"{cpu.replace(' ', '-')}.json"
    path.write_text(json.dumps({
        "earshot": "0.1.0",
        "material": "",
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor() or platform.machine(),
                    "cpu": cpu},
        "runs": runs,
    }), encoding="utf-8")
    return str(path)


def test_scoreboard_keeps_two_machines_apart(tmp_path):
    """The defect this file exists for: a throughput measured on an M1 Max and
    one measured on an M2 must not render as the same machine."""
    files = [_results(tmp_path, "Apple M1 Max"), _results(tmp_path, "Apple M2")]
    table = report.scoreboard(files)
    assert "Apple M1 Max" in table
    assert "Apple M2" in table


def test_the_recorded_machine_names_the_cpu():
    """A result carries the CPU, not just the instruction set."""
    machine = json.loads(report.as_json([]))["machine"]
    assert machine["cpu"] == report.cpu_name()
    assert machine["cpu"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="sysctl is macOS only")
def test_the_cpu_name_is_what_the_system_reports():
    """Measured against the system, not against a string we chose."""
    expected = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert report.cpu_name() == expected
