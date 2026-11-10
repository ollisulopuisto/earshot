from pathlib import Path
import os
import pytest
import numpy as np
import soundfile as sf
from earshot.testing import assert_engine_contract
from earshot.engines.vst3 import VST3Engine


@pytest.fixture(scope="module")
def built_bridge_vst3():
    pytest.importorskip("pedalboard", reason="the vst3 engine needs the 'vst3' extra")
    plugin_path = Path("plugin/earshot_bridge/build/Release/EarshotBridge.vst3")
    if not plugin_path.exists():
        pytest.skip("EarshotBridge.vst3 is not built")
    return str(plugin_path)


def test_bridge_plugin_built_and_satisfies_contract(built_bridge_vst3):
    engine = VST3Engine(built_bridge_vst3)
    assert engine.name == "EarshotBridge"
    assert_engine_contract(engine)


def test_bridge_parameters_and_restore_workflow(built_bridge_vst3):
    import pedalboard

    plugin = pedalboard.load_plugin(built_bridge_vst3)
    assert "capture_selection" in plugin.parameters
    assert "restore_with_earshot" in plugin.parameters

    rate = 48000
    n = 4800
    data = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / rate)).astype(np.float32)
    block = data.reshape(1, -1)

    # 1. Passthrough mode
    out = plugin.process(block, rate, reset=True)[0]
    assert np.allclose(data, out, atol=1e-4)

    # 2. Capture selection
    plugin.capture_selection = True
    plugin.process(block, rate, reset=False)
    plugin.capture_selection = False

    # 3. Trigger restore by processing a block with restore_with_earshot = True
    plugin.restore_with_earshot = True
    plugin.process(block, rate, reset=False)

    # In temporary bridge dir, selection_in.wav should have been written
    in_wav = "/tmp/earshot_bridge/selection_in.wav"
    assert os.path.exists(in_wav)
    captured, in_rate = sf.read(in_wav)
    assert in_rate == rate
    assert len(captured) == n
    assert np.allclose(data, captured, atol=1e-4)
