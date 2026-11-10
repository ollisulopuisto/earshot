from pathlib import Path
import pytest
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



# The capture/restore workflow is tested without a prebuilt plugin bundle by
# plugin/earshot_bridge/tests/bridge_state_test.cpp (run in the tests workflow).
# It awaits the worker, checks the actual per-instance WAV directory, and covers
# ownership transfer, delayed completion and host seeks. This optional test above
# only checks a locally installed VST3 wrapper against the Python engine contract.
