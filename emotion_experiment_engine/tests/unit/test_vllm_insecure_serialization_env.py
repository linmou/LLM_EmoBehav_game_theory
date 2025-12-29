"""Tests for setting vLLM insecure serialization env early.

Responsible files:
- emotion_experiment_engine/emotion_experiment_series_runner.py

Purpose:
- RepE's vLLM hook registration uses `collective_rpc` with callables, which
  requires `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on vLLM v1 (secure serializer
  otherwise rejects function objects).
- This env must be set before vLLM is imported (vLLM caches envs).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_runner_sets_vllm_allow_insecure_serialization_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """I am starting with a failing test. This is the Red phase."""
    from emotion_experiment_engine.emotion_experiment_series_runner import (
        MemoryExperimentSeriesRunner,
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "experiment_name: test",
                "models: ['dummy-model']",
                "emotions: ['anger']",
                "intensities: [1.0]",
                "benchmarks:",
                "  - name: game_theory_decision",
                "    task_type: Prisoners_Dilemma",
                "    sample_limit: 1",
                "loading_config:",
                "  additional_vllm_kwargs:",
                "    attention_backend: TRITON_ATTN",
                "repe_eng_config:",
                "  data_dir: data/stimulus/repeng_generated_images_with_text",
                "output_dir: " + str(tmp_path / "out"),
            ]
        )
    )

    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)

    MemoryExperimentSeriesRunner(config_path=str(cfg_path), dry_run=True)
    assert os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") == "1"

