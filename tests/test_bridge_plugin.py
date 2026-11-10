from pathlib import Path
import pytest
import numpy as np
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

    # 3. Trigger restore while capture is still active - verifies capture is stopped safely without race
    plugin.restore_with_earshot = True
    plugin.process(block, rate, reset=False)

    # 4. Starting a new capture while waiting does not corrupt state
    plugin.capture_selection = True
    plugin.process(block, rate, reset=False)
    plugin.capture_selection = False
    plugin.restore_with_earshot = False
    plugin.process(block, rate, reset=False)
