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


def test_the_margin_the_loader_accepts_actually_moves_the_edge():
    """``router:<engine>@<margin>`` was parsed, stored and recorded in the
    result notes, but the edge detector read the module constant instead, so
    every margin behaved as 40 dB. A parameter that is accepted and silently
    ignored puts a number in the record that never reached the code.
    """
    from types import SimpleNamespace

    import numpy as np
    from scipy import signal as sig

    from earshot.engines import router as R

    rate, n = 48000, 48000 * 3
    rng = np.random.default_rng(0)
    band_limited = sig.sosfilt(
        sig.butter(8, 8000, "low", fs=rate, output="sos"),
        rng.standard_normal(n) * 0.1,
    ).astype(np.float32)
    tone = (0.05 * np.sin(2 * np.pi * 14000 * np.arange(n) / rate)).astype(np.float32)

    class Inner:
        name = "marker"

        def process(self, audio, rate):
            return np.asarray(audio, dtype=np.float32) + tone

    inner = SimpleNamespace(engine=Inner())

    edges = {}
    for margin in (10.0, 90.0):
        engine = R.RouterEngine(inner, margin)
        engine.process(band_limited, rate)
        edges[margin] = float(np.median(engine.edges))

    # A stricter margin admits less as content, so the edge sits lower.
    assert edges[10.0] < edges[90.0], (
        f"margin ignored: 10 dB and 90 dB both put the edge at {edges[10.0]:.0f} Hz"
    )


def test_two_margins_are_two_names():
    """A bench run with two thresholds produced 111 rows under one name, so
    the means merged three engines and the record could not say which
    threshold produced which number. The default keeps its bare name, or
    every result measured before this would stop being comparable.
    """
    from types import SimpleNamespace

    from earshot.engines import router as R

    inner = SimpleNamespace(engine=SimpleNamespace(name="lavasr"))
    assert R.RouterEngine(inner, R.CLIFF_DB).name == "router(lavasr)"
    assert R.RouterEngine(inner, 60.0).name != R.RouterEngine(inner, 25.0).name
    assert "60" in R.RouterEngine(inner, 60.0).name


def test_silence_does_not_drag_the_edge_up():
    """Gated pauses were pulling the crossover into the band it should fill.

    Silent frames are parked at the ceiling so the router will not invent a
    voice into them, and then the edge is smoothed in time. The smoothing
    averaged those ceiling values into the speech frames beside them, so the
    more silence a recording had, the higher the router put its crossover.

    It matters most on exactly the material this is for. Measured on the one
    genuine 7.5 kHz call in hand, which is 40.7 per cent silent by frame: the
    raw edge on speech frames is 7195 Hz, the smoothed edge 9754 Hz, and
    smoothing over speech frames alone 7162 Hz. The router was declining to
    fill 2.6 kHz of the band that carries presence.
    """
    import numpy as np
    from scipy import signal as sig

    from earshot import probes
    from earshot.engines import router as R

    rate = 48000
    cutoff = 7000.0

    # Band-limited speech in bursts, with hard digital silence between them,
    # which is what a gated call actually looks like.
    speech = probes.default_material(rate, 12.0)
    speech = sig.sosfilt(
        sig.butter(10, cutoff, "low", fs=rate, output="sos"), speech
    ).astype(np.float32)
    gated = np.zeros_like(speech)
    burst, pause = int(0.8 * rate), int(0.8 * rate)
    at = 0
    while at + burst < len(speech):
        gated[at:at + burst] = speech[at:at + burst]
        at += burst + pause

    from types import SimpleNamespace

    class Inner:
        name = "inner"

        def process(self, audio, rate):
            return np.asarray(audio, dtype=np.float32)

    engine = R.RouterEngine(SimpleNamespace(engine=Inner()), R.CLIFF_DB)
    engine.process(gated, rate)
    edges = np.array(engine.edges)

    ceiling = min(R.EDGE_CEILING_HZ, rate / 2 * 0.95)
    acting = edges[edges < ceiling * 0.999]
    assert len(acting) > 0, "the router declined to act on band-limited speech"

    # The crossover should sit near the real cutoff, not be dragged toward
    # the ceiling by the pauses.
    assert np.median(acting) < cutoff * 1.5, (
        f"crossover at {np.median(acting):.0f} Hz for a {cutoff:.0f} Hz "
        f"cutoff — the silence pulled it up"
    )
