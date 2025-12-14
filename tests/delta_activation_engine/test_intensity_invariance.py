# Responsible file: delta_activation_engine/runner.py
# Purpose: Check that zero intensity yields ~0 delta and doubling intensity increases ||delta||.

import io
import os
import numpy as np

from delta_activation_engine.backends import BaseBackend
from delta_activation_engine.config import load_job_config_from_yaml
from delta_activation_engine.pipelines.runner import run_job


class LinearFakeBackend(BaseBackend):
    def __init__(self, hidden_dim=6):
        self.h = hidden_dim
        self.base = np.arange(self.h, dtype=np.float32) * 0.0  # zero for simplicity
        self.dir = np.arange(1, self.h + 1, dtype=np.float32)  # [1,2,3,...]

    def get_repr(self, prompts, *, steered, emotion=None, intensity=None):
        if not steered:
            return self.base.copy()
        k = float(intensity or 0.0)
        return self.base + k * self.dir


def test_intensity_scaling(tmp_path):
    yaml = f"""
model_path: /models/DUMMY
emotions: [anger]
intensities: [0.0, 1.0, 2.0]
output_dir: {tmp_path.as_posix()}
loading_config: {{ model_path: /models/DUMMY, max_model_len: 4096 }}
repe_eng_config: {{ control_method: reading_vec, block_name: decoder_block, rep_token: "<REP>", data_dir: data/stimulus/text/, n_difference: 8, direction_method: mean-diff, emotions: [anger] }}
"""
    cfg = load_job_config_from_yaml(io.StringIO(yaml))
    out_dir = run_job(cfg, LinearFakeBackend(hidden_dim=6))

    d0 = np.load(os.path.join(out_dir, "deltas", "emotion=anger_int=0.0.npz"))["vector"]
    d1 = np.load(os.path.join(out_dir, "deltas", "emotion=anger_int=1.0.npz"))["vector"]
    d2 = np.load(os.path.join(out_dir, "deltas", "emotion=anger_int=2.0.npz"))["vector"]

    # Norm zero at intensity 0
    assert np.linalg.norm(d0) == 0.0
    # Monotonic increase with intensity
    assert np.linalg.norm(d1) < np.linalg.norm(d2)
