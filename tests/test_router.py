"""The router has one job: find where the input stops.

If it finds the wrong edge it either replaces a voice that was fine or
declines to fix one that was not, and neither failure announces itself. So
the edge is what gets tested, on signals whose true edge is known.
"""

import numpy as np
import pytest
from scipy import signal as sig

from earshot import engines
from earshot.engines.router import RouterEngine
from earshot.testing import assert_engine_contract

RATE = 48000


def speechlike(seconds=4.0, rate=RATE, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    voiced = 0.3 * sig.square(2 * np.pi * np.cumsum(np.full(n, 130.0)) / rate, duty=0.08)
    out = np.zeros(n)
    for lo, hi, g in ((200, 900, 1.0), (900, 2400, 0.5), (2400, 7000, 0.25)):
        sos = sig.butter(2, [lo / (rate / 2), hi / (rate / 2)], btype="band", output="sos")
        out += g * sig.sosfilt(sos, voiced)
    out += 10 ** (-70 / 20) * rng.normal(0, 1, n)
    return (out / (np.abs(out).max() + 1e-9) * 0.3).astype(np.float32)


def edge_of(x, rate=RATE):
    router = RouterEngine(engines.load("passthrough"))
    router.process(x, rate)
    return float(np.median(router.edges))


def test_the_edge_is_reported_for_inspection():
    """The edge is the whole design, so it has to be readable from outside."""
    router = RouterEngine(engines.load("passthrough"))
    router.process(speechlike(2.0), RATE)
    assert len(router.edges) > 10 and all(e > 0 for e in router.edges)


def brickwall(x, cut, rate=RATE):
    """Zero everything above `cut`, which is what a codec does.

    A Butterworth is not a cliff — an eighth-order filter is still only 60 dB
    down more than an octave past its corner — and the router exists for
    material a codec emptied, not material a filter sloped.
    """
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(len(x), 1 / rate) > cut] = 0
    return np.fft.irfft(spectrum, len(x)).astype(np.float32)


def test_a_codec_style_cut_is_found_where_it_is():
    """The whole premise: find where the material stops."""
    x = speechlike()
    for cut in (3400.0, 8000.0):
        found = edge_of(brickwall(x, cut))
        assert found == pytest.approx(cut, rel=0.15), f"cut {cut} -> found {found}"


def test_a_gentle_roll_off_reads_high_and_that_is_the_safe_direction():
    """A dull microphone must not be mistaken for a band-limited one.

    A filtered roll-off still has content in its transition band, and the
    router reads the edge above the corner because of it — measured 4.7 kHz
    for a 3.4 kHz eighth-order cut. That errs towards keeping the input,
    which is the direction to err in: declining to improve a recording costs
    nothing, replacing a voice that was fine costs the voice.
    """
    x = speechlike()
    gentle = sig.sosfiltfilt(sig.butter(8, 3400 / (RATE / 2), output="sos"), x)
    assert edge_of(gentle.astype(np.float32)) > 3400.0


def test_full_band_material_is_left_alone():
    """No edge worth acting on means the engine does not get to invent."""
    rng = np.random.default_rng(1)
    wide = (speechlike() + 0.01 * rng.normal(0, 1, int(4 * RATE))).astype(np.float32)
    assert edge_of(wide) > 12000.0


def test_silence_is_not_given_a_voice():
    """A passage with no speech must not have speech invented into it."""

    class Shouts:
        name = "shouts"

        def process(self, audio, rate):
            rng = np.random.default_rng(0)
            return (np.asarray(audio) + 0.2 * rng.normal(0, 1, len(audio))).astype(
                "float32"
            )

    router = RouterEngine(engines.Loaded(Shouts()))
    quiet = np.zeros(RATE * 3, dtype=np.float32)
    out = router.process(quiet, RATE)
    level = 20 * np.log10(np.sqrt(np.mean(out.astype(np.float64) ** 2)) + 1e-12)
    assert level < -40, f"silence came back at {level:.1f} dBFS"


def test_below_the_edge_the_input_survives():
    """The point of routing rather than replacing."""

    class Wipes:
        name = "wipes"

        def process(self, audio, rate):
            return np.zeros_like(np.asarray(audio))

    x = sig.sosfiltfilt(sig.butter(8, 4000 / (RATE / 2), output="sos"),
                        speechlike()).astype(np.float32)
    out = RouterEngine(engines.Loaded(Wipes())).process(x, RATE)
    band = sig.butter(4, [300 / (RATE / 2), 2000 / (RATE / 2)], btype="band", output="sos")
    kept = np.corrcoef(sig.sosfilt(band, x), sig.sosfilt(band, out))[0, 1]
    assert kept > 0.8, f"input was not preserved below the edge (r={kept:.2f})"


def test_the_router_keeps_the_contract():
    assert_engine_contract(engines.load("router:passthrough").engine)


def test_the_margin_can_be_set_and_is_checked():
    assert engines.load("router:passthrough@12").engine.margin_db == 12.0
    with pytest.raises(engines.EngineError):
        engines.load("router:passthrough@loud")
