"""Which device the PyTorch engines run on.

Every torch engine was pinned to the CPU, which on a Colab GPU wastes the
GPU: UniverSR takes about 12 minutes per 8 s on an M1 Max CPU.
``EARSHOT_DEVICE`` chooses; unset, CUDA is used where it exists. The Apple
GPU is never chosen on its own: UniverSR on MPS was 11 s per pass against
15 s on the CPU, and the first attempt sat waiting on the GPU for over ten
minutes, so it is opt-in.
"""

import sys
import types

import pytest

from earshot import engines


def test_the_variable_decides(monkeypatch):
    monkeypatch.setenv("EARSHOT_DEVICE", "cuda:1")
    assert engines.torch_device() == "cuda:1"


def test_a_device_torch_would_not_know_is_refused(monkeypatch):
    monkeypatch.setenv("EARSHOT_DEVICE", "gpu")
    with pytest.raises(engines.EngineError) as caught:
        engines.torch_device()
    assert "EARSHOT_DEVICE" in str(caught.value)


@pytest.mark.parametrize("cuda, expected", [(True, "cuda"), (False, "cpu")])
def test_unset_means_cuda_where_there_is_one(monkeypatch, cuda, expected):
    monkeypatch.delenv("EARSHOT_DEVICE", raising=False)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert engines.torch_device() == expected


def test_unset_without_torch_is_the_cpu(monkeypatch):
    monkeypatch.delenv("EARSHOT_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises
    assert engines.torch_device() == "cpu"
