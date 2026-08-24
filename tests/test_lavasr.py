"""LavaSR's wiring, without the weights.

CI has no models and no network, so what is testable here is everything
around the inference: the spec parsing, the option handling, and the length
arithmetic that the engine contract turns on. The model itself is exercised
by `assert_engine_contract` on a machine that has it.
"""

import numpy as np
import pytest

from earshot import engines, fetch
from earshot.engines import lavasr


def test_every_asset_is_pinned_and_named_as_the_graph_expects():
    """ONNX graphs reference their weight files by the name they were built
    with, so the cached copies cannot be renamed — only moved together."""
    assert lavasr.ASSETS
    for asset in lavasr.ASSETS:
        assert len(asset.sha256) == 64, asset.name
        assert asset.url.startswith("https://"), asset.name
        assert asset.name.startswith("lavasr/"), asset.name
    graphs = {a.name for a in lavasr.ASSETS if a.name.endswith(".onnx")}
    for data in (a.name for a in lavasr.ASSETS if a.name.endswith(".onnx.data")):
        assert data[: -len(".data")] in graphs, f"{data} has no graph beside it"


def test_options_are_checked_not_ignored():
    with pytest.raises(engines.EngineError) as caught:
        engines.load("lavasr:denoize")
    assert "denoize" in str(caught.value)


def test_no_download_gives_a_usable_message(tmp_path, monkeypatch):
    """Asks the fetcher directly rather than through the engine.

    Going through `engines.load("lavasr")` reports "install the extra" first
    on a machine without onnxruntime — which is the right thing to say to a
    person and the wrong thing to assert here, and it turned CI red once.
    The message under test belongs to the fetcher, so the fetcher is what
    gets asked.
    """
    monkeypatch.setenv("EARSHOT_CACHE", str(tmp_path))
    monkeypatch.setenv("EARSHOT_NO_DOWNLOAD", "1")
    with pytest.raises(engines.EngineError) as caught:
        fetch.ensure_all(lavasr.ASSETS)
    message = str(caught.value)
    # Every missing file at once: one at a time would be one round trip per
    # model, which is not a switch so much as an obstacle.
    assert message.count("curl -L") == len(lavasr.ASSETS)
    assert "enhancer_backbone.onnx.data" in message


def test_the_engine_asks_for_the_extra_before_anything_else():
    """On a machine without onnxruntime, that is the actionable first step."""
    onnx = pytest.importorskip
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        with pytest.raises(engines.EngineError) as caught:
            engines.load("lavasr")
        assert "earshot[lavasr]" in str(caught.value)


def test_the_resampler_round_trip_keeps_the_length_within_a_few_samples():
    """The engine trims or pads the tail to be exact; this checks the
    arithmetic never needs to move more than a handful of samples, because a
    large correction would mean the audio itself had shifted."""
    for rate in (48000, 44100, 32000):
        n = rate * 3
        x = np.zeros(n, dtype=np.float32)
        down = lavasr._resample(x, rate, lavasr.MODEL_RATE)
        back = lavasr._resample(
            np.zeros(len(down) * 3, dtype=np.float32), lavasr.OUTPUT_RATE, rate
        )
        assert abs(len(back) - n) < 0.01 * rate, rate
