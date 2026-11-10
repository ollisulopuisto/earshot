"""Reuse work for recovery before/after scores without changing the metric."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy import signal

from earshot import degrade, engines, metrics, probes


def original_log_spectral_distance(
    a: np.ndarray,
    b: np.ndarray,
    rate: int,
    low: float = 0.0,
    high: float = 0.0,
    match_level: bool = True,
) -> float:
    """Original un-hoisted definition of log_spectral_distance."""
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


@pytest.mark.parametrize("rate,n", [(8000, 1024), (16000, 2049), (48000, 48001)])
@pytest.mark.parametrize(
    "low,high", [(0.0, 0.0), (3400.0, 16000.0), (100.0, 1000.0)]
)
@pytest.mark.parametrize("match_level", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_plan_and_wrapper_match_original(rate, n, low, high, match_level, dtype):
    rng = np.random.default_rng(21)
    reference = rng.normal(size=n).astype(dtype)
    test_signals = [
        rng.normal(size=n).astype(dtype),
        (reference * 0.8).astype(dtype),
        np.zeros(n, dtype=dtype),
    ]
    plan = metrics.LogSpectralDistancePlan(
        reference, rate, low=low, high=high, match_level=match_level
    )
    for sig in test_signals:
        expected = original_log_spectral_distance(
            sig, reference, rate, low=low, high=high, match_level=match_level
        )
        actual_plan = plan.score(sig)
        actual_func = metrics.log_spectral_distance(
            sig, reference, rate, low=low, high=high, match_level=match_level
        )
        if np.isnan(expected):
            assert np.isnan(actual_plan)
            assert np.isnan(actual_func)
        else:
            assert actual_plan == pytest.approx(expected, rel=1e-12, abs=1e-12)
            assert actual_func == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_plan_owns_derived_reference_state_and_reuses_stft():
    rng = np.random.default_rng(22)
    reference = rng.normal(size=2048).astype(np.float32)
    sig1 = rng.normal(size=2048).astype(np.float32)
    sig2 = rng.normal(size=2048).astype(np.float32)
    expected1 = original_log_spectral_distance(sig1, reference, 16000)
    expected2 = original_log_spectral_distance(sig2, reference, 16000)

    with patch.object(signal, "stft", wraps=signal.stft) as stft_mock:
        plan = metrics.LogSpectralDistancePlan(reference, 16000)
        # Mutating reference caller buffer in place should not corrupt the plan
        reference[:] = 0.0
        score1 = plan.score(sig1)
        score2 = plan.score(sig2)

    assert score1 == pytest.approx(expected1, rel=1e-12, abs=1e-12)
    assert score2 == pytest.approx(expected2, rel=1e-12, abs=1e-12)
    # Hoisting exists to compute the reference STFT once per plan, so the
    # call count is the reuse contract: a fourth call means it regressed.
    assert stft_mock.call_count == 3


def test_empty_audio_scores_nan_like_an_empty_band():
    """An absent signal has no measurable distance; NaN is how the metric says
    'no measurement', like an empty frequency band, while 0 would read as a
    perfect restoration.
    """
    assert np.isnan(metrics.log_spectral_distance(np.zeros(0), np.zeros(0), 16000))
    assert np.isnan(metrics.log_spectral_distance(np.zeros(1024), np.zeros(0), 16000))
    assert np.isnan(metrics.log_spectral_distance(np.zeros(0), np.zeros(1024), 16000))
    plan = metrics.LogSpectralDistancePlan(np.zeros(0), 16000)
    assert np.isnan(plan.score(np.zeros(0)))
    with pytest.raises(ValueError, match="shape"):
        plan.score(np.zeros(1024))


def test_validation_and_shape_guards():
    plan = metrics.LogSpectralDistancePlan(np.ones(1024), 16000)
    assert plan.score(np.ones(1024)) == pytest.approx(0.0, abs=1e-12)

    # The bench compares sample against sample, so a misaligned signal is a
    # contract violation (raise), never a score.
    for bad_sig in [np.zeros(1023), np.zeros(1025), np.zeros((1024, 1))]:
        with pytest.raises(ValueError, match="shape"):
            plan.score(bad_sig)

    # A plan built on a malformed reference or rate cannot measure anything;
    # failing at construction keeps the error at the call that caused it.
    for bad_ref in [np.zeros((1024, 1)), np.zeros((2, 2))]:
        with pytest.raises(ValueError, match="reference"):
            metrics.LogSpectralDistancePlan(bad_ref, 16000)

    for bad_rate in [0, -16000, float("nan")]:
        with pytest.raises(ValueError, match="rate"):
            metrics.LogSpectralDistancePlan(np.zeros(1024), bad_rate)

    # An empty band leaves no bins to compare. NaN rather than zero, so the
    # score can never read as a perfect restoration - but the shape guard
    # still fires first, because misalignment stays a contract violation
    # even when there is nothing to measure.
    nan_plan = metrics.LogSpectralDistancePlan(np.ones(1024), 16000, low=10000, high=5000)
    assert np.isnan(nan_plan.score(np.ones(1024)))
    assert np.isnan(metrics.log_spectral_distance(np.ones(1024), np.ones(1024), 16000, low=10000, high=5000))
    with pytest.raises(ValueError, match="shape"):
        nan_plan.score(np.ones(512))


def test_probe_uses_one_plan_for_recovery():
    clean = probes.default_material(48000, 1.0)
    with patch.object(
        metrics, "LogSpectralDistancePlan", wraps=metrics.LogSpectralDistancePlan
    ) as plan:
        with patch.object(
            metrics, "log_spectral_distance", side_effect=AssertionError("not reused")
        ):
            run = probes.run_all(
                engines.load("passthrough"),
                clean,
                48000,
                degrade.by_name("narrowband-voip"),
            )
            assert abs(run.value("recovery", "gained")) < 0.01
            assert plan.call_count == 1
