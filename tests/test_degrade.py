"""The damage must be damage, and nothing else.

Every recipe has to keep the sample count and the alignment, because the
bench compares sample against sample. A degradation that shifted the signal
would be scored as damage no engine could ever undo, and the whole bench
would quietly measure the wrong thing.
"""

import numpy as np
import pytest

from earshot import degrade, probes

RATE = 48000


@pytest.fixture(scope="module")
def clean():
    return probes.default_material(RATE, 4.0)


@pytest.mark.parametrize("recipe", [r for r in degrade.RECIPES if not r.needs],
                         ids=lambda r: r.name)
def test_damage_keeps_the_sample_count(recipe, clean):
    out = recipe.apply(clean, RATE)
    assert len(out) == len(clean), recipe.name
    assert np.all(np.isfinite(out)), recipe.name


def test_band_limit_removes_the_band_it_says():
    from scipy import signal
    x = probes.default_material(RATE, 3.0)
    y = degrade.band_limit(x, RATE, low=0, high=4000)
    f, px = signal.welch(x, RATE, nperseg=4096)
    _, py = signal.welch(y, RATE, nperseg=4096)
    above = (f > 6000) & (f < 16000)
    below = (f > 500) & (f < 3000)
    assert 10 * np.log10(py[above].mean() / px[above].mean()) < -30
    assert abs(10 * np.log10(py[below].mean() / px[below].mean())) < 3


def test_clean_is_actually_untouched(clean):
    assert np.array_equal(degrade.by_name("clean").apply(clean, RATE), clean)


def test_noise_hits_the_requested_snr(clean):
    noisy = degrade.noise(clean, RATE, snr_db=20.0)
    added = noisy - clean
    speech = degrade._speech_rms(clean, RATE)
    snr = 20 * np.log10(speech / (np.sqrt(np.mean(added**2)) + 1e-12))
    assert 18 < snr < 22


def test_autogain_flattens_the_dynamics(clean):
    from earshot import metrics
    before = metrics.dynamics(clean, RATE)
    after = metrics.dynamics(degrade.autogain(clean, RATE), RATE)
    # This is the point of the recipe: it destroys the distance between
    # speech and silence, which is the information no engine can get back.
    assert after.range_db < before.range_db - 5


def test_unknown_recipe_names_the_known_ones():
    with pytest.raises(KeyError) as caught:
        degrade.by_name("ei-ole")
    assert "narrowband-voip" in str(caught.value)


def test_a_voip_recipe_leaves_the_ceiling_it_claims():
    """A call cannot carry what the codec never transmitted.

    ``wideband-voip`` says "8 kHz ceiling", and ``band_limit`` delivered one:
    on real material 14 kHz fell to -147 dB relative to the speech band. The
    next step then added broadband noise across the whole spectrum, refilling
    the band the recipe had just emptied to about -24 dB, and the recipe
    stopped modelling a call. The router declined to engage on it — correctly,
    the signal really did have energy to 24 kHz — and LavaSR scored +7.92 dB
    for removing noise that no telephone band would have carried.

    Measured as a tilt rather than an absolute level, so the assertion is
    about what the recipe did and not about how bright the material was.
    """
    from scipy import signal

    x = probes.default_material(RATE, 4.0)
    y = degrade.by_name("wideband-voip").apply(x, RATE)

    f, px = signal.welch(x, RATE, nperseg=4096)
    _, py = signal.welch(y, RATE, nperseg=4096)
    speech = (f >= 300) & (f <= 3400)
    above = (f >= 12000) & (f <= 20000)

    def tilt(p):
        return 10 * np.log10(p[above].mean() / p[speech].mean())

    moved = tilt(py) - tilt(px)
    assert moved < -20, (
        f"the band above the ceiling was refilled: tilt moved {moved:+.1f} dB, "
        f"expected a drop of at least 20 dB"
    )


def _speech_with_room_tone(seconds: float = 6.0, tone_db: float = -70.0):
    """Speech bursts separated by room tone, shaped like the measured files.

    ``probes.default_material`` spans only 40 dB between its loudest and
    quietest window, so it cannot exercise a gate calibrated against real
    recordings, where speech sits at -27 to -44 dBFS and the floor at -65 to
    -92. This builds that gap explicitly rather than tuning the gate down to
    meet the synthetic material.
    """
    speech = probes.default_material(RATE, seconds)
    rng = np.random.default_rng(0)
    tone = rng.normal(0.0, 10 ** (tone_db / 20), len(speech)).astype(np.float32)
    out = tone.copy()
    burst = int(0.9 * RATE)
    pause = int(0.6 * RATE)
    at = 0
    while at + burst < len(speech):
        out[at:at + burst] = speech[at:at + burst]
        at += burst + pause
    return out


def test_the_gate_makes_the_silence_digital():
    """The defect that separates real platform audio from a studio track.

    Measured over every file in hand: studio recordings contain no run of
    eight or more zero samples in 224 minutes, while every remote-platform
    file is 8.0 to 57.8 per cent exact zero — panu-recording-4 holds 482,449
    separate runs in twelve minutes, and the one genuine 7.5 kHz call is
    57.8 per cent zero with gaps up to 21 seconds. No recipe modelled it, and
    the VoIP recipes added noise where the real damage removes it.
    """
    x = _speech_with_room_tone()
    assert (x == 0.0).mean() < 0.01, "the source already has digital silence"

    y = degrade.gate(x, RATE)

    assert len(y) == len(x)
    zero = (y == 0.0).mean()
    assert 0.05 < zero < 0.75, f"gated {zero:.1%}, expected the measured 8-58 % range"

    # Where it did not gate, it must not have touched the signal: this is a
    # gate, not a compressor.
    kept = y != 0.0
    assert np.allclose(y[kept], x[kept]), "the gate altered the signal it passed"

    # The speech itself must survive. A gate that eats words is a different
    # degradation and would be scored as damage no engine could undo.
    loud = np.abs(x) > np.percentile(np.abs(x), 99) * 0.5
    assert (y[loud] != 0.0).mean() > 0.95, "the gate cut into the speech"


def test_the_platform_recipe_has_the_signature_it_was_built_from():
    """Both halves of the measured signature, or the recipe is a guess again."""
    from scipy import signal

    x = _speech_with_room_tone(8.0)
    y = degrade.by_name("platform-upload").apply(x, RATE)

    zero = (y == 0.0).mean()
    assert 0.05 < zero < 0.75, f"digital silence {zero:.1%}, measured range is 8-58 %"

    f, px = signal.welch(x, RATE, nperseg=4096)
    _, py = signal.welch(y, RATE, nperseg=4096)
    stop = (f >= 17000) & (f <= 22000)
    passband = (f >= 500) & (f <= 3000)
    assert 10 * np.log10(py[stop].mean() / px[stop].mean()) < -30, "no ceiling"
    assert abs(10 * np.log10(py[passband].mean() / px[passband].mean())) < 3, \
        "the speech band was not left alone"
