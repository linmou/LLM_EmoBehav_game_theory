"""
Tests for `neuro_manipulation/utils.py::load_model_only`.

Purpose: When we load a HF model on CUDA first (for rep-reader extraction) and
then initialize vLLM, vLLM's multiprocessing workers must use `spawn` (not fork)
or CUDA initialization can fail in the worker processes.
"""

import os
import unittest
from unittest.mock import MagicMock


class TestVllmWorkerMultiprocMethod(unittest.TestCase):
    def test_load_model_only_sets_vllm_worker_multiproc_method_spawn(self):
        # Responsible file: neuro_manipulation/utils.py
        # Purpose: avoid "CUDA driver initialization failed" in vLLM workers.
        from neuro_manipulation import utils as nm_utils

        os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)

        fake_loading_config = MagicMock()
        fake_loading_config.to_vllm_kwargs.return_value = {
            "model": "fake-model",
            "tensor_parallel_size": None,
        }

        # neuro_manipulation.utils is a package that re-exports functions from a
        # separately loaded module ("utils_module"). Patch through function globals.
        g = nm_utils.load_model_only.__globals__
        fake_llm = MagicMock(return_value=MagicMock())
        with unittest.mock.patch.dict(
            g,
            {
                "LLM": fake_llm,
                "get_optimal_tensor_parallel_size": MagicMock(return_value=1),
                "is_awq_model": MagicMock(return_value=False),
            },
            clear=False,
        ):
            nm_utils.load_model_only(
                model_name_or_path="fake-model",
                from_vllm=True,
                loading_config=fake_loading_config,
            )

        self.assertEqual(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"), "spawn")
