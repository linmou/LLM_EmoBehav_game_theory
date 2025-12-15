"""
Responsible: delta_activation_engine/pipelines/runner.py
Purpose: Orchestration for baseline delta activation computation (non-chat).
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Dict, List

import numpy as np

from ..backends.base import BaseBackend
from ..config.job import DeltaActivationJobConfig
from ..io.files import ensure_dir, save_json, save_npz_vector
from ..prompts.probes_texts import get_generic_probes


def _hash_probes(probes: List[str]) -> str:
    h = hashlib.sha256()
    for p in probes:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def run_job(cfg: DeltaActivationJobConfig, backend: BaseBackend) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_base = os.path.basename(cfg.model_path.rstrip("/")) or "model"
    out_dir = os.path.join(cfg.output_dir, f"{model_base}_{timestamp}")
    ensure_dir(out_dir)

    templates = get_generic_probes()
    probe_hash = _hash_probes(templates)
    probes = templates

    baseline_vec = backend.get_repr(probes, steered=False)
    save_npz_vector(os.path.join(out_dir, "baseline.npz"), baseline_vec)

    metadata: Dict[str, object] = {
        "model_path": cfg.model_path,
        "emotions": cfg.emotions,
        "intensities": cfg.intensities,
        "probe_hash": probe_hash,
        "timestamp": timestamp,
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
                probes, steered=True, emotion=emo, intensity=float(it)
            )
            delta = steered_vec - baseline_vec
            fname = f"emotion={emo}_int={float(it)}.npz"
            save_npz_vector(os.path.join(deltas_dir, fname), delta)

    return out_dir

