"""
Tests for `neuro_manipulation/repe/rep_control_vllm_hook.py::RepControlVLLMHook`.

Purpose: vLLM v1 stores tensor-parallel size under llm_engine.vllm_config.parallel_config.
RepControlVLLMHook should detect that (instead of defaulting to tp_size=1).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


class TestRepControlVllmHookTpSizeDetection(unittest.TestCase):
    def test_tp_size_detected_from_vllm_config_parallel_config(self):
        from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook

        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=4)
        )
        llm_engine = SimpleNamespace(
            collective_rpc=MagicMock(return_value=[True]),
            vllm_config=vllm_config,
        )
        model = SimpleNamespace(llm_engine=llm_engine)

        hook = RepControlVLLMHook(
            model=model,
            tokenizer=MagicMock(),
            layers=[0],
            block_name="decoder_block",
            control_method="reading_vec",
            tensor_parallel_size=1,
        )

        self.assertEqual(hook.tp_size, 4)
