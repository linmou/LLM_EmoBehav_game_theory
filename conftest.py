"""Global pytest configuration.

This repo has many optional heavyweight dependencies (notably PyTorch). In some
environments, importing the real `torch` binary can hard-crash the interpreter.

This stub is opt-in: set `USE_TORCH_STUB=1` to force a lightweight `torch` stub.
"""

from __future__ import annotations

import os
import sys
import types
import importlib.machinery


def _install_torch_stub() -> None:
    if os.environ.get("USE_TORCH_STUB") != "1":
        return

    if "torch" in sys.modules:
        return

    torch_module = types.ModuleType("torch")
    torch_module.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

    # Minimal dtype sentinels used in tests/config.
    torch_module.float32 = object()
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_module.int64 = object()

    def manual_seed(seed: int) -> None:  # pragma: no cover - trivial stub
        del seed

    torch_module.manual_seed = manual_seed

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def get_device_properties(index: int):
            raise RuntimeError("CUDA not available in torch stub")

    torch_module.cuda = _Cuda

    # torch.utils.data stubs
    utils_module = types.ModuleType("torch.utils")
    utils_module.__spec__ = importlib.machinery.ModuleSpec("torch.utils", loader=None)
    data_module = types.ModuleType("torch.utils.data")
    data_module.__spec__ = importlib.machinery.ModuleSpec("torch.utils.data", loader=None)

    class _TorchDataset:  # pragma: no cover - placeholder
        def __iter__(self):
            return iter(())

    class _TorchDataLoader:  # pragma: no cover - placeholder
        def __init__(self, dataset, batch_size: int | None = 1, shuffle: bool = False, collate_fn=None):
            self.dataset = dataset
            self.batch_size = batch_size or 1
            self.shuffle = shuffle
            self.collate_fn = collate_fn

        def __iter__(self):
            data_iter = None
            if hasattr(self.dataset, "__iter__"):
                data_iter = iter(self.dataset)
            elif hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__"):
                data_iter = (self.dataset[i] for i in range(len(self.dataset)))
            else:
                data_iter = iter(())

            batch = []
            for item in data_iter:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield self._collate(batch)
                    batch = []
            if batch:
                yield self._collate(batch)

        def _collate(self, items):
            if self.collate_fn is not None:
                return self.collate_fn(items)
            if self.batch_size == 1:
                return items[0]
            return list(items)

    data_module.Dataset = _TorchDataset
    data_module.DataLoader = _TorchDataLoader
    utils_module.data = data_module
    torch_module.utils = utils_module

    # torch.nn stubs (imports should succeed; tests usually mock behavior).
    nn_module = types.ModuleType("torch.nn")
    nn_module.__spec__ = importlib.machinery.ModuleSpec("torch.nn", loader=None)

    class _Module:  # pragma: no cover - placeholder
        pass

    nn_module.Module = _Module
    torch_module.nn = nn_module

    # torch.distributed stub
    dist_module = types.ModuleType("torch.distributed")
    dist_module.__spec__ = importlib.machinery.ModuleSpec("torch.distributed", loader=None)

    def init_process_group(*args, **kwargs) -> None:  # pragma: no cover
        del args, kwargs

    dist_module.init_process_group = init_process_group
    torch_module.distributed = dist_module

    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = utils_module
    sys.modules["torch.utils.data"] = data_module
    sys.modules["torch.nn"] = nn_module
    sys.modules["torch.distributed"] = dist_module


_install_torch_stub()
