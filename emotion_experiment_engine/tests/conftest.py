"""Test configuration for emotion_experiment_engine tests."""

import sys
import types


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return

    torch_module = types.ModuleType("torch")
    utils_module = types.ModuleType("torch.utils")
    data_module = types.ModuleType("torch.utils.data")

    class _TorchDataset:  # type: ignore[override]
        pass

    data_module.Dataset = _TorchDataset
    utils_module.data = data_module
    torch_module.utils = utils_module

    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = utils_module
    sys.modules["torch.utils.data"] = data_module


_install_torch_stub()
