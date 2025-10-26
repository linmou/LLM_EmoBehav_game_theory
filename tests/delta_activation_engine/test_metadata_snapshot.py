# Responsible file: delta_activation_engine/runner.py
# Purpose: Ensure metadata.json contains a full config snapshot and backend metadata.

import io
import json
import os

from delta_activation_engine.backends import BaseBackend
from delta_activation_engine.config import load_job_config_from_yaml
from delta_activation_engine.pipelines.runner import run_job


class MetaBackend(BaseBackend):
    def get_repr(self, prompts, *, steered, emotion=None, intensity=None):
        import numpy as np
        return np.zeros((3,), dtype=np.float32)

    def get_run_metadata(self) -> dict:
        return {"foo": "bar"}


def test_metadata_includes_config_and_backend(tmp_path):
    yaml = f"""
model_path: /models/DUMMY
emotions: [anger]
intensities: [0.0]
output_dir: {tmp_path.as_posix()}
loading_config: {{ model_path: /models/DUMMY, max_model_len: 4096 }}
repe_eng_config: {{ control_method: reading_vec, block_name: decoder_block, rep_token: "<REP>", data_dir: data/stimulus/text/, n_difference: 8, direction_method: mean-diff, emotions: [anger] }}
"""
    cfg = load_job_config_from_yaml(io.StringIO(yaml))
    out_dir = run_job(cfg, MetaBackend())

    with open(os.path.join(out_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "job_config" in meta
    assert meta["job_config"]["model_path"] == "/models/DUMMY"
    assert meta.get("backend_metadata", {}).get("foo") == "bar"
