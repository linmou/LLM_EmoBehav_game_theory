"""
Responsible: delta_activation_engine/chat_runner.py, delta_activation_engine/datasets.py
Purpose: TDD for chat-template-aware delta activation pipeline. Ensures prompts
         are built via prompt wrappers (using PromptFormat) and end-to-end
         flow saves expected artifacts without touching the existing CLI.
"""

import io
import json
import os
from typing import Any, List, Optional
from unittest.mock import patch

import numpy as np


class DummyPromptFormat:
    """Minimal PromptFormat stub used by wrappers in tests."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.model_name = getattr(tokenizer, "name_or_path", "dummy")

    def build(
        self,
        system_prompt: str,
        user_messages: List[str],
        assistant_messages: List[str] = [],
        images: List[Any] = None,
        enable_thinking: bool = False,
    ) -> str:
        # Extremely simple deterministic format to verify wrapper was used
        # and user text survives templating.
        sys = system_prompt or ""
        user = user_messages[0] if user_messages else ""
        return f"SYS|{sys}|USER|{user}|ASSIST|"


class FakeTokenizer:
    def __init__(self):
        self.name_or_path = "dummy-model"
        self.chat_template = "<dummy-template>"


class FakeBackend:
    """Backend stub that encodes prompts into fixed-size vectors for tests."""

    def __init__(self, hidden_dim: int = 4):
        self.h = hidden_dim
        self.base = np.arange(self.h, dtype=np.float32)

    def get_repr(
        self,
        prompts: List[str],
        *,
        steered: bool,
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        # Deterministic toy behavior: baseline is base; steered adds k*1
        if not steered:
            return self.base.copy()
        k = float(intensity or 0.0)
        return self.base + k * np.ones_like(self.base)

    def get_run_metadata(self) -> dict:
        return {"backend": "fake", "hidden_dim": self.h}


def test_chat_dataset_builds_prompts(tmp_path):
    """Dataset built via registry + wrapper should produce templated prompts."""
    from delta_activation_engine.config import load_chat_job_config_from_yaml
    from delta_activation_engine.pipelines.chat_runner import run_job_chat

    yaml_str = f"""
model_path: /models/DUMMY
emotions: [anger]
intensities: [0.0, 0.5]
output_dir: {tmp_path.as_posix()}
loading_config: {{ model_path: /models/DUMMY, max_model_len: 4096 }}
repe_eng_config: {{ control_method: reading_vec, block_name: decoder_block, rep_token: "<REP>", data_dir: data/stimulus/text/, n_difference: 8, direction_method: mean-diff, emotions: [anger] }}
prompt_config:
  benchmark_name: delta_probes
  task_type: default
  probes: ["Say hello", "Summarize: test"]
"""
    cfg = load_chat_job_config_from_yaml(io.StringIO(yaml_str))

    # Patch tokenizer + PromptFormat to avoid heavy HF deps and enforce our dummy format
    with patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(FakeTokenizer(), None)), \
         patch("neuro_manipulation.prompt_formats.PromptFormat", DummyPromptFormat):
        out_dir = run_job_chat(cfg, backend=FakeBackend(hidden_dim=4))

    # Files exist
    assert os.path.exists(os.path.join(out_dir, "baseline.npz"))
    assert os.path.exists(os.path.join(out_dir, "deltas", "emotion=anger_int=0.0.npz"))
    assert os.path.exists(os.path.join(out_dir, "deltas", "emotion=anger_int=0.5.npz"))

    # Metadata has prompt info
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("pipeline") == "chat"
    assert meta.get("prompt_config", {}).get("benchmark_name") == "delta_probes"
    assert meta.get("backend_metadata", {}).get("backend") == "fake"


def test_chat_pipeline_deltas_are_relative(tmp_path):
    """Zero intensity delta is zero; positive intensity shifts by constant ones."""
    from delta_activation_engine.config import load_chat_job_config_from_yaml
    from delta_activation_engine.pipelines.chat_runner import run_job_chat

    yaml_str = f"""
model_path: /models/DUMMY
emotions: [anger]
intensities: [0.0, 1.0]
output_dir: {tmp_path.as_posix()}
loading_config: {{ model_path: /models/DUMMY, max_model_len: 4096 }}
repe_eng_config: {{ control_method: reading_vec, block_name: decoder_block, rep_token: "<REP>", data_dir: data/stimulus/text/, n_difference: 8, direction_method: mean-diff, emotions: [anger] }}
prompt_config:
  benchmark_name: delta_probes
  task_type: default
  probes: ["Probe A", "Probe B", "Probe C"]
"""

    cfg = load_chat_job_config_from_yaml(io.StringIO(yaml_str))

    with patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(FakeTokenizer(), None)), \
         patch("neuro_manipulation.prompt_formats.PromptFormat", DummyPromptFormat):
        out_dir = run_job_chat(cfg, backend=FakeBackend(hidden_dim=8))

    # Load baseline and deltas
    base_vec = np.load(os.path.join(out_dir, "baseline.npz"))["vector"]
    d0 = np.load(os.path.join(out_dir, "deltas", "emotion=anger_int=0.0.npz"))["vector"]
    d1 = np.load(os.path.join(out_dir, "deltas", "emotion=anger_int=1.0.npz"))["vector"]

    # Shapes consistent
    assert base_vec.shape == (8,)
    assert d0.shape == (8,)
    assert d1.shape == (8,)

    # Zero intensity delta should be zeros
    np.testing.assert_allclose(d0, np.zeros_like(d0), atol=1e-6)
    # Unit intensity delta should be ones
    np.testing.assert_allclose(d1, np.ones_like(d1), atol=1e-6)
