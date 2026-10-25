"""WPE dereverberation: the classical baseline the research survey put first.

Weighted prediction error predicts the late reverberation from the past of
the signal and subtracts it, linearly; it cannot remove the direct path, so
it should not take speech out with the echo — the failure DeepFilterNet
showed on podcast rooms (blind pick: the damaged take beat every setting).
"""

import numpy as np
import pytest

from earshot import degrade, engines

RATE = 48000


def _engine():
    try:
        return engines.load("wpe").engine
    except engines.EngineError as exc:
        pytest.skip(f"wpe unavailable: {exc}")


def _speech(seconds=4.0):
    from earshot import probes

    return probes.default_material(RATE, seconds)


def test_it_keeps_the_contract():
    from earshot.testing import assert_engine_contract

    assert_engine_contract(_engine())


def test_it_takes_out_reverberation_not_the_voice():
    """On real speech in a room with a realistic direct-to-reverberant ratio
    (room-near, +6 dB): closer to the dry original, and not further from its
    speaker. Measured on five EARS excerpts: +2.17 dB, speaker +0.024.
    WPE needs speech that starts and stops — on the steady synthetic test
    material it had nothing to predict from and moved nothing."""
    from pathlib import Path

    import soundfile as sf

    from earshot import metrics

    path = Path(__file__).parent.parent / "out/colab-calls/out/kuuntelu-bwe-ears/01-puhdas/00-alkuperainen.wav"
    if not path.exists():
        pytest.skip("needs a real clean EARS excerpt in out/")
    dry, rate = sf.read(path, dtype="float32")
    wet = degrade.by_name("room-near").apply(dry, rate)
    out = _engine().process(wet, rate)
    gained = (metrics.log_spectral_distance(wet, dry, rate, 100, 8000)
              - metrics.log_spectral_distance(out, dry, rate, 100, 8000))
    assert gained > 1.0
    from earshot import speaker

    if speaker.available():
        before = float(np.dot(speaker.embed(dry, rate), speaker.embed(wet, rate)))
        after = float(np.dot(speaker.embed(dry, rate), speaker.embed(out, rate)))
        assert after >= before - 0.005


def test_dry_input_is_left_nearly_alone():
    """Not untouched — the gate is what keeps clean audio away from it."""
    dry = _speech()
    out = _engine().process(dry, RATE)
    change = 10 * np.log10(np.mean((out - dry) ** 2) / np.mean(dry**2))
    assert change < -6.0


def test_options_are_checked():
    with pytest.raises(engines.EngineError):
        engines.load("wpe:taps=lots")
