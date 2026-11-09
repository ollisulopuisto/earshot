"""Reuse work for tonal before/after scores without changing the metric."""

from unittest.mock import patch

import numpy as np
import pytest

from earshot import degrade, engines, metrics, probes


def original_line_residue(error, reference, rate, fundamental=50.0, top=6000.0):
    """Independent full-mask definition, before the reusable plan."""
    e = np.asarray(error, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    window = np.hanning(len(e))
    spectrum = np.abs(np.fft.rfft(e * window)) ** 2
    freqs = np.fft.rfftfreq(len(e), 1 / rate)
    lines = np.zeros_like(freqs, dtype=bool)
    for k in range(1, int(min(top, rate / 2 * 0.95) // fundamental) + 1):
        if k * fundamental < min(top, rate / 2 * 0.95):
            lines |= np.abs(freqs - k * fundamental) <= 1.0 + 0.3 * k
    speech = np.sum(np.abs(np.fft.rfft(r * window)) ** 2) + 1e-30
    return float(10 * np.log10(spectrum[lines].sum() / speech + 1e-30))


@pytest.mark.parametrize("rate,n", [(8000, 31), (16000, 4096), (48000, 48001)])
@pytest.mark.parametrize(
    "fundamental,top", [(50.0, 6000.0), (60.0, 2400.0), (50.0, 40.0)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_plan_and_wrapper_match_original(rate, n, fundamental, top, dtype):
    rng = np.random.default_rng(15)
    reference = rng.normal(size=n).astype(dtype)
    errors = [rng.normal(size=n).astype(dtype), np.zeros(n, dtype=dtype)]
    plan = metrics.LineResiduePlan(reference, rate, fundamental, top)
    for error in errors:
        expected = original_line_residue(error, reference, rate, fundamental, top)
        assert plan.score(error) == expected
        assert (
            metrics.line_residue(error, reference, rate, fundamental, top) == expected
        )


def test_plan_owns_derived_reference_state_and_reuses_fft():
    reference = np.linspace(-0.5, 0.5, 1024)
    error = np.cos(np.arange(1024))
    expected = original_line_residue(error, reference, 8000)
    with patch.object(np.fft, "rfft", wraps=np.fft.rfft) as fft:
        with patch.object(np, "hanning", wraps=np.hanning) as window:
            plan = metrics.LineResiduePlan(reference, 8000)
            reference[:] = 0
            assert plan.score(error) == expected
            assert plan.score(error) == expected
    assert fft.call_count == 3  # one reference FFT and one per error
    assert window.call_count == 1


def test_silent_reference_and_length_contract():
    plan = metrics.LineResiduePlan(np.zeros(128), 8000)
    assert plan.score(np.zeros(128)) == -300.0
    for error in [np.zeros(127), np.zeros(129), np.zeros((128, 1))]:
        with pytest.raises(ValueError, match="shape"):
            plan.score(error)
    for reference in [np.zeros(0), np.zeros((128, 1))]:
        with pytest.raises(ValueError):
            metrics.LineResiduePlan(reference, 8000)


def test_probe_uses_one_plan_for_both_scores_and_none_without_tonal_damage():
    clean = probes.default_material(48000, 1.0)
    with patch.object(
        metrics, "LineResiduePlan", wraps=metrics.LineResiduePlan
    ) as plan:
        with patch.object(
            metrics, "line_residue", side_effect=AssertionError("not reused")
        ):
            run = probes.run_all(
                engines.load("passthrough"), clean, 48000, degrade.by_name("hum")
            )
            assert run.value("tonal", "removed") == 0.0
            assert plan.call_count == 1
            probes.run_all(
                engines.load("passthrough"), clean, 48000, degrade.by_name("clean")
            )
            assert plan.call_count == 1
