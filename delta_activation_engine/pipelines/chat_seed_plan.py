"""
Responsible: delta_activation_engine/pipelines/chat_seed_plan.py
Purpose: Build seeded chat job configs for multiple models without mutating the base config.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List

from ..config.chat_job import DeltaActivationChatJobConfig, PromptingConfig


def _clone_prompt_config(pc: PromptingConfig) -> PromptingConfig:
    return PromptingConfig(
        benchmark_name=pc.benchmark_name,
        task_type=pc.task_type,
        probes=list(pc.probes) if pc.probes is not None else None,
        probe_source=pc.probe_source,
        enable_thinking=pc.enable_thinking,
    )


def build_seeded_chat_jobs(
    base_cfg: DeltaActivationChatJobConfig,
    model_paths: Iterable[str],
    seeds: Iterable[int],
    *,
    output_root: str,
) -> List[DeltaActivationChatJobConfig]:
    jobs: List[DeltaActivationChatJobConfig] = []
    for model_path in model_paths:
        for seed in seeds:
            loading_cfg = dict(base_cfg.loading_config)
            loading_cfg["model_path"] = model_path
            loading_cfg["seed"] = int(seed)
            repe_cfg = deepcopy(base_cfg.repe_eng_config)
            repe_cfg["emotion_data_seed"] = int(seed)
            jobs.append(
                DeltaActivationChatJobConfig(
                    model_path=model_path,
                    emotions=list(base_cfg.emotions),
                    intensities=[float(x) for x in base_cfg.intensities],
                    output_dir=output_root,
                    loading_config=loading_cfg,
                    repe_eng_config=repe_cfg,
                    prompt_config=_clone_prompt_config(base_cfg.prompt_config),
                )
            )
    return jobs
