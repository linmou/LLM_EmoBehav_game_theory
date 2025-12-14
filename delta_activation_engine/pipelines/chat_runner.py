"""
Responsible: delta_activation_engine/pipelines/chat_runner.py
Purpose: Chat-template-aware delta activation pipeline using wrappers + dataset.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional

from ..backends.base import BaseBackend
from ..backends.hf import HFBackend
from ..config.chat_job import DeltaActivationChatJobConfig
from ..config.job import DeltaActivationJobConfig
from ..datasets.probes import DeltaProbesDataset
from ..io.files import ensure_dir, save_json, save_npz_vector
from ..prompts.probes_texts import get_generic_probes
from ..prompts.wrappers import DeltaProbesPromptWrapper


def _hash_texts(texts: List[str]) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def _collect_prompts(dataset) -> List[str]:
    prompts: List[str] = []
    for i in range(len(dataset)):
        rec = dataset[i]
        prompts.append(rec["prompt"])  # Dataset returns dict with 'prompt'
    return prompts


def run_job_chat(cfg: DeltaActivationChatJobConfig, *, backend: Optional[BaseBackend] = None) -> str:
    from neuro_manipulation.utils import load_tokenizer_only
    from neuro_manipulation.prompt_formats import PromptFormat
    from emotion_experiment_engine.data_models import BenchmarkConfig

    tokenizer, _ = load_tokenizer_only(
        model_name_or_path=cfg.model_path,
        expand_vocab=False,
        auto_load_multimodal=True,
    )
    prompt_format = PromptFormat(tokenizer)

    bcfg = BenchmarkConfig(
        name=cfg.prompt_config.benchmark_name,
        task_type=cfg.prompt_config.task_type,
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    if cfg.prompt_config.probes is not None:
        probes = list(cfg.prompt_config.probes)
    elif (cfg.prompt_config.probe_source or "").lower() == "generic":
        probes = get_generic_probes()
    else:
        probes = get_generic_probes()

    wrapper = DeltaProbesPromptWrapper(
        prompt_format,
        user_messages="Please provide your answer.",
        enable_thinking=bool(cfg.prompt_config.enable_thinking)
        if cfg.prompt_config.enable_thinking is not None
        else False,
        system_prompt="",
    )

    dataset = DeltaProbesDataset(
        config=bcfg,
        prompt_wrapper=wrapper,
        max_context_length=None,
        tokenizer=tokenizer,
        truncation_strategy="right",
        probes=probes,
    )

    final_prompts = _collect_prompts(dataset)
    probe_hash = _hash_texts(final_prompts)

    if backend is None:
        shim_cfg = DeltaActivationJobConfig(
            model_path=cfg.model_path,
            emotions=cfg.emotions,
            intensities=cfg.intensities,
            output_dir=cfg.output_dir,
            loading_config=cfg.loading_config,
            repe_eng_config=cfg.repe_eng_config,
        )
        backend = HFBackend(shim_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_base = os.path.basename(cfg.model_path.rstrip("/")) or "model"
    out_dir = os.path.join(cfg.output_dir, "chat", f"{model_base}_{timestamp}")
    ensure_dir(out_dir)

    baseline_vec = backend.get_repr(final_prompts, steered=False)
    save_npz_vector(os.path.join(out_dir, "baseline.npz"), baseline_vec)

    metadata: Dict[str, object] = {
        "pipeline": "chat",
        "model_path": cfg.model_path,
        "emotions": cfg.emotions,
        "intensities": cfg.intensities,
        "probe_hash": probe_hash,
        "timestamp": timestamp,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "prompt_config": {
            "benchmark_name": cfg.prompt_config.benchmark_name,
            "task_type": cfg.prompt_config.task_type,
            "probe_source": cfg.prompt_config.probe_source,
            "num_prompts": len(final_prompts),
            "enable_thinking": cfg.prompt_config.enable_thinking,
        },
        "job_config": {
            "model_path": cfg.model_path,
            "emotions": cfg.emotions,
            "intensities": cfg.intensities,
            "output_dir": cfg.output_dir,
            "loading_config": cfg.loading_config,
            "repe_eng_config": cfg.repe_eng_config,
        },
        "backend_metadata": backend.get_run_metadata(),
    }
    save_json(os.path.join(out_dir, "metadata.json"), metadata)

    deltas_dir = os.path.join(out_dir, "deltas")
    ensure_dir(deltas_dir)
    for emo in cfg.emotions:
        for it in cfg.intensities:
            steered_vec = backend.get_repr(
                final_prompts, steered=True, emotion=emo, intensity=float(it)
            )
            delta = steered_vec - baseline_vec
            fname = f"emotion={emo}_int={float(it)}.npz"
            save_npz_vector(os.path.join(deltas_dir, fname), delta)

    return out_dir

