"""
Responsible: tests/delta_activation_engine/test_chat_seed_plan.py
Purpose: Validate seeded chat run planning produces one config per model/seed without mutating the base config.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from delta_activation_engine.config.chat_job import (
    DeltaActivationChatJobConfig,
    PromptingConfig,
)
from delta_activation_engine.pipelines.chat_seed_plan import build_seeded_chat_jobs


def _make_base_cfg() -> DeltaActivationChatJobConfig:
    return DeltaActivationChatJobConfig(
        model_path="base-model",
        emotions=["anger", "happiness"],
        intensities=[0.0, 1.0],
        output_dir="results/delta_activations",
        loading_config={"model_path": "base-model", "seed": 0},
        repe_eng_config={"foo": "bar"},
        prompt_config=PromptingConfig(
            benchmark_name="delta_probes",
            task_type="default",
            probes=None,
            probe_source="generic",
            enable_thinking=False,
        ),
    )


def test_build_seeded_chat_jobs_creates_full_grid_without_mutation() -> None:
    base_cfg = _make_base_cfg()
    models = [
        os.path.expandvars("${USER_HOME}/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct").replace("${USER_HOME}", "/home/jjl7137"),
        os.path.expandvars("${USER_HOME}/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct").replace("${USER_HOME}", "/home/jjl7137"),
        os.path.expandvars("${USER_HOME}/huggingface_models/Qwen/Qwen2.5-3B-Instruct").replace("${USER_HOME}", "/home/jjl7137"),
    ]
    seeds = list(range(20))
    jobs = build_seeded_chat_jobs(base_cfg, models, seeds, output_root="results/delta_activations/chat")

    assert len(jobs) == 60
    # First job should use first model and seed 0
    assert jobs[0].model_path == models[0]
    assert jobs[0].loading_config["seed"] == 0
    assert jobs[0].repe_eng_config["emotion_data_seed"] == 0
    # Last job should use last model and last seed
    assert jobs[-1].model_path == models[-1]
    assert jobs[-1].loading_config["seed"] == seeds[-1]
    assert jobs[-1].repe_eng_config["emotion_data_seed"] == seeds[-1]
    # Base config should remain unchanged
    assert base_cfg.model_path == "base-model"
    assert base_cfg.loading_config["seed"] == 0
    assert base_cfg.output_dir == "results/delta_activations"
    assert "emotion_data_seed" not in base_cfg.repe_eng_config
