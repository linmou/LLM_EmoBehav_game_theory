# Responsible file: delta_activation_engine/runner.py
# Purpose: End-to-end small run with a fake backend; checks files, shapes, metadata.

import io
import json
import os
import shutil
import tempfile
from typing import List, Optional

import numpy as np

from delta_activation_engine.config import DeltaActivationJobConfig, load_job_config_from_yaml
from delta_activation_engine.pipelines.runner import run_job
from delta_activation_engine.backends import BaseBackend


class FakeBackend(BaseBackend):
    def __init__(self, hidden_dim: int = 4):
        self.h = hidden_dim
        # fixed base vector for determinism
        self.base = np.arange(self.h, dtype=np.float32)
        self.dir = np.ones(self.h, dtype=np.float32)

    def get_repr(
        self,
        prompts: List[str],
        *,
        steered: bool,
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        if not steered:
            return self.base.copy()
        k = float(intensity or 0.0)
        return self.base + k * self.dir


def test_e2e_small(tmp_path):
    # YAML config (minimal)
    yaml_str = f"""
model_path: /models/DUMMY
emotions: [anger]
intensities: [0.0, 1.0]
output_dir: {tmp_path.as_posix()}
loading_config: {{ model_path: /models/DUMMY, max_model_len: 4096 }}
repe_eng_config: {{ control_method: reading_vec, block_name: decoder_block, rep_token: "<REP>", data_dir: data/stimulus/text/, n_difference: 8, direction_method: mean-diff, emotions: [anger] }}
"""
    cfg = load_job_config_from_yaml(io.StringIO(yaml_str))

    out_dir = run_job(cfg, FakeBackend(hidden_dim=4))

    # Baseline exists
    assert os.path.exists(os.path.join(out_dir, "baseline.npz"))

    # Metadata exists and contains expected keys
    meta_path = os.path.join(out_dir, "metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for key in ["model_path", "emotions", "intensities", "probe_hash", "timestamp"]:
        assert key in meta

    # Deltas for both intensities
    d0 = np.load(os.path.join(out_dir, "deltas", "emotion=anger_int=0.0.npz"))
    d1 = np.load(os.path.join(out_dir, "deltas", "emotion=anger_int=1.0.npz"))
    v0 = d0["vector"]
    v1 = d1["vector"]

    assert v0.shape == (4,)
    assert v1.shape == (4,)
    # Zero intensity delta should be zeros
    np.testing.assert_allclose(v0, np.zeros_like(v0), atol=1e-6)
    # Unit intensity delta should be ones
    np.testing.assert_allclose(v1, np.ones_like(v1), atol=1e-6)
