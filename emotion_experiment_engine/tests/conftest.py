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


def _install_vllm_stub() -> None:
    if "vllm" in sys.modules:
        return

    vllm_module = types.ModuleType("vllm")

    class _StubLLM:  # minimal placeholder for tests
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_module.LLM = _StubLLM
    class _StubSamplingParams:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_module.SamplingParams = _StubSamplingParams
    sys.modules["vllm"] = vllm_module


_install_vllm_stub()


def _install_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    openai_module = types.ModuleType("openai")

    class _StubChatCompletions:
        def create(self, *args, **kwargs):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"label": "refuse"}'))])

    class _StubChat:
        def __init__(self):
            self.completions = _StubChatCompletions()

    class _StubOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _StubChat()

    openai_module.OpenAI = _StubOpenAI
    sys.modules["openai"] = openai_module


_install_openai_stub()
