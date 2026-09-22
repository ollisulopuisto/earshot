"""The deterministic repairs: hum, plosives, clipping, and gated silence.

Each is tested for the contract, for doing nothing where there is nothing to
do, and for doing its one job on the damage it exists for. The synthetic
material here proves the mechanism, not the quality; quality is measured by
the bench on real voices.
"""

import numpy as np

from earshot import degrade, engines, probes
from earshot.engines.declip import DeclipEngine
from earshot.engines.dehum import DehumEngine
from earshot.engines.keepzero import KeepZeroEngine
from earshot.engines.plosive import DeplosiveEngine, HighpassEngine
from earshot.testing import assert_engine_contract

RATE = 48000


def _db(a):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)) + 1e-12)


def test_the_repairs_keep_the_contract():
    for engine in (DehumEngine(50.0), DehumEngine(None), DeplosiveEngine(),
                   HighpassEngine(80.0), DeclipEngine(),
                   KeepZeroEngine(engines.load("passthrough"))):
        assert_engine_contract(engine)


def test_the_repairs_are_registered():
    for spec, name in (("dehum:50", "dehum(50)"), ("dehum", "dehum(auto)"),
                       ("dehum:50@2", "dehum(50@2Hz)"), ("deplosive", "deplosive"),
                       ("deplosive:200", "deplosive(200Hz)"),
                       ("highpass:80", "highpass(80Hz)"), ("declip", "declip"),
                       ("keepzero:passthrough", "keepzero(passthrough)")):
        assert engines.load(spec).engine.name == name


def test_dehum_removes_a_drifting_hum_and_nothing_else():
    clean = probes.default_material(RATE, 6.0)
    hummed = degrade.hum(clean, RATE, level_db=-20.0, harmonics=3)
    engine = DehumEngine(50.0)
    out = engine.process(hummed, RATE)
    assert engine.found["harmonics"], "found no hum in a hummed signal"
    assert _db(out - clean) < _db(hummed - clean) - 15


def test_dehum_leaves_a_recording_without_hum_bit_for_bit():
    # Not probes.default_material: its pulse train sweeps through 450 Hz and
    # holds still there long enough to look like a ninth harmonic, which is
    # a fair reading of a synthetic signal and not what this test is about.
    rng = np.random.default_rng(3)
    t = np.arange(6 * RATE) / RATE
    clean = (0.2 * np.sin(2 * np.pi * 173 * t + 2 * np.sin(2 * np.pi * 0.9 * t))
             + 0.02 * rng.normal(0, 1, len(t))).astype(np.float32)
    engine = DehumEngine(50.0)
    assert np.array_equal(engine.process(clean, RATE), clean)
    assert engine.found["harmonics"] == []


def test_declip_only_touches_the_plateaus():
    t = np.arange(RATE) / RATE
    x = (0.8 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    clipped = degrade.clip(x, RATE, headroom_db=-3.0)
    engine = DeclipEngine()
    out = engine.process(clipped, RATE)
    assert engine.repaired > 0
    plateau = np.abs(clipped) >= np.abs(clipped).max() * 0.999
    assert np.array_equal(out[~plateau], clipped[~plateau])
    # Redrawn towards the true peak, never back inside the rail.
    assert np.all(np.abs(out[plateau]) >= np.abs(clipped[plateau]) - 1e-7)
    assert _db(out - x) < _db(clipped - x) - 6


def test_declip_leaves_an_unclipped_file_alone():
    clean = probes.default_material(RATE, 3.0)
    assert np.array_equal(DeclipEngine().process(clean, RATE), clean)


def test_deplosive_cuts_a_low_burst_and_keeps_the_rest():
    t = np.arange(3 * RATE) / RATE
    voice = 0.2 * np.sin(2 * np.pi * 440 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    burst = np.zeros_like(t)
    start = int(1.5 * RATE)
    tail = np.arange(int(0.2 * RATE)) / RATE
    burst[start:start + len(tail)] = 0.5 * np.exp(-tail / 0.04) * np.sin(2 * np.pi * 45 * tail)
    x = (voice + burst).astype(np.float32)
    engine = DeplosiveEngine()
    out = engine.process(x, RATE)
    assert engine.events, "no event found around a 45 Hz burst"
    assert _db((out - voice)[start:start + len(tail)]) < _db(burst[start:start + len(tail)]) - 6
    # A second away from the burst the output is the input.
    assert np.allclose(out[: RATE // 2], x[: RATE // 2], atol=1e-6)


def test_keepzero_restores_the_gate_after_an_inventing_engine():
    class Inventor:
        name = "inventor"

        def process(self, audio, rate):
            return np.asarray(audio, dtype=np.float32) + 0.01

    x = probes.default_material(RATE, 2.0).copy()
    x[RATE // 2 : RATE] = 0.0
    wrapped = KeepZeroEngine(engines.Loaded(Inventor()))
    out = wrapped.process(x, RATE)
    assert np.all(out[RATE // 2 : RATE] == 0.0)
    assert np.allclose(out[: RATE // 4], x[: RATE // 4] + 0.01, atol=1e-6)


def test_the_new_recipes_do_what_they_say():
    clean = probes.default_material(RATE, 6.0)
    hummed = degrade.by_name("hum").apply(clean, RATE)
    from scipy import signal

    f, p = signal.welch(hummed - clean, RATE, nperseg=RATE)
    assert f[np.argmax(p)] in (49.0, 50.0, 51.0)
    clipped = degrade.by_name("clipped").apply(clean, RATE)
    assert np.abs(clipped).max() < np.abs(clean).max() * 0.6
