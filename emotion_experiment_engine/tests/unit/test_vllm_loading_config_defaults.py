# Tests for emotion_experiment_engine/data_models.py: VLLMLoadingConfig should provide safe vLLM worker extension defaults.

import unittest
import tempfile
from pathlib import Path


class TestVllmLoadingConfigDefaults(unittest.TestCase):
    def test_to_vllm_kwargs_sets_worker_extension_cls_by_default(self):
        from emotion_experiment_engine.data_models import VLLMLoadingConfig

        cfg = VLLMLoadingConfig(
            model_path="/tmp/model",
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            max_model_len=1024,
            enforce_eager=True,
            quantization=None,
            trust_remote_code=True,
            dtype="bfloat16",
            seed=0,
            disable_custom_all_reduce=False,
            additional_vllm_kwargs={},
        )
        kwargs = cfg.to_vllm_kwargs()
        self.assertIn("worker_extension_cls", kwargs)
        self.assertEqual(
            kwargs["worker_extension_cls"],
            "neuro_manipulation.repe.vllm_worker_extension.NMRepControlWorkerExtension",
        )

    def test_to_vllm_kwargs_sets_hf_overrides_dtype_for_local_paths(self):
        # emotion_experiment_engine/data_models.py: ensure local model paths don't trigger
        # vLLM HF Hub safetensors metadata probing (which rejects filesystem paths).
        from emotion_experiment_engine.data_models import VLLMLoadingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "fake_model"
            model_dir.mkdir(parents=True)

            cfg = VLLMLoadingConfig(
                model_path=str(model_dir),
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                max_model_len=1024,
                enforce_eager=True,
                quantization=None,
                trust_remote_code=True,
                dtype="float16",
                seed=0,
                disable_custom_all_reduce=False,
                additional_vllm_kwargs={},
            )
            kwargs = cfg.to_vllm_kwargs()
            self.assertIn("hf_overrides", kwargs)
            self.assertEqual(kwargs["hf_overrides"]["dtype"], "float16")

    def test_to_vllm_kwargs_merges_hf_overrides_when_provided(self):
        # emotion_experiment_engine/data_models.py: preserve user hf_overrides and only
        # add dtype when missing, to keep local-path loads quiet and deterministic.
        from emotion_experiment_engine.data_models import VLLMLoadingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "fake_model"
            model_dir.mkdir(parents=True)

            cfg = VLLMLoadingConfig(
                model_path=str(model_dir),
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                max_model_len=1024,
                enforce_eager=True,
                quantization=None,
                trust_remote_code=True,
                dtype="bfloat16",
                seed=0,
                disable_custom_all_reduce=False,
                additional_vllm_kwargs={"hf_overrides": {"architectures": ["X"]}},
            )
            kwargs = cfg.to_vllm_kwargs()
            self.assertEqual(kwargs["hf_overrides"]["architectures"], ["X"])
            self.assertEqual(kwargs["hf_overrides"]["dtype"], "bfloat16")
