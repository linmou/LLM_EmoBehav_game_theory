# Tests that RepControlVLLMHook handles vLLM engine failures gracefully by raising a clear exception.
from pathlib import Path
from types import SimpleNamespace
import pytest

# Minimal fake LLM engine that will raise the EngineDeadError-like exception when collective_rpc is called.
class _DeadEngine:
    class EngineDeadError(Exception):
        pass

    def collective_rpc(self, *args, **kwargs):
        raise self.EngineDeadError("engine dead for test")


class _FakeLLM:
    def __init__(self):
        self.llm_engine = _DeadEngine()
        # mimic parallel_config for tp_size
        self.llm_engine.parallel_config = SimpleNamespace(tensor_parallel_size=1)


def test_engine_dead_error_raises_runtime_with_context(monkeypatch):
    from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook

    model = _FakeLLM()
    tokenizer = SimpleNamespace()
    # Inject fake hook function import dependency
    monkeypatch.setattr("neuro_manipulation.repe.rep_control_vllm_hook.hook_fn_rep_control", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="engine dead for test"):
        RepControlVLLMHook(model, tokenizer, layers=[0], block_name="decoder_block", control_method="reading_vec")
