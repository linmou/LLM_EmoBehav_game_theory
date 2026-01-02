"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Compare sample-level per-layer delta activations induced by
         (1) an emotion steering vector and
         (2) a PD-defection steering vector,
         on the Prisoner's Dilemma dataset (default: test split).

This module is intentionally KISS:
- Pure helpers are unit-tested (layer mapping + cosine).
- Heavy model/dataset code lives in the CLI runner below.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def build_output_dir(
    *,
    output_root: Path,
    run_id: str,
    model_path: str,
    emotion: str,
    split_seed: int,
) -> Path:
    """
    Output layout:
      <output_root>/<run_id>/<model_name>/<emotion>/seed_<split_seed>/

    run_id is intended to be a datetime string so the top-level folder is a stable identifier.
    """
    return (
        Path(output_root)
        / str(run_id)
        / Path(str(model_path)).name
        / str(emotion)
        / f"seed_{int(split_seed)}"
    )


def middle_third_layers(num_layers: int) -> List[int]:
    start = num_layers // 3
    end = (2 * num_layers) // 3
    return list(range(start, end))


def progress(it: Iterable[Any], *, total: int | None = None, desc: str | None = None) -> Iterable[Any]:
    """
    Tiny wrapper around tqdm with a safe fallback.
    Kept as a pure helper so unit tests can ensure it doesn't break runs.
    """
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return it
    return tqdm(it, total=total, desc=desc)


def repreader_key_for_layer(*, layer: int, num_layers: int) -> int:
    """
    RepReader direction keys are negative: -num_layers..-1.
    Mapping used in repo: key = layer - num_layers.
    """
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"layer out of range: layer={layer} num_layers={num_layers}")
    return int(layer - num_layers)


def cosine_per_layer(delta_a: np.ndarray, delta_b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute cosine similarity per sample, per layer.

    Args:
        delta_a: (batch, layers, hidden)
        delta_b: (batch, layers, hidden)
        eps: if norm_prod < eps => NaN
    """
    if delta_a.shape != delta_b.shape:
        raise ValueError(f"shape mismatch: {delta_a.shape} vs {delta_b.shape}")
    if delta_a.ndim != 3:
        raise ValueError(f"expected 3D arrays, got {delta_a.shape}")

    a = np.asarray(delta_a, dtype=np.float32)
    b = np.asarray(delta_b, dtype=np.float32)
    dot = np.sum(a * b, axis=-1, dtype=np.float32)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = na * nb
    out = np.empty_like(dot, dtype=np.float32)
    valid = denom >= float(eps)
    np.divide(dot, denom, out=out, where=valid)
    out[~valid] = np.nan
    return out


def write_csv_outputs(
    *,
    out_dir: Path,
    item_ids: Sequence[int],
    prompts: Sequence[str],
    intensities: Sequence[float],
    controlled_layers: Sequence[int],
    cosines: np.ndarray,
    delta_norms_anger: np.ndarray,
    delta_norms_pd: np.ndarray,
) -> None:
    """
    Write CSVs for downstream analysis without duplicating the full prompt per row.

    - samples.csv: one row per sample (item_id, prompt)
    - cosines.csv: one row per (sample, intensity, layer) with cosine + delta norms
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(item_ids) != len(prompts):
        raise ValueError("item_ids and prompts must have same length")
    if cosines.ndim != 3:
        raise ValueError(f"cosines must be (n_int, n_samples, n_layers), got {cosines.shape}")
    if cosines.shape != delta_norms_anger.shape or cosines.shape != delta_norms_pd.shape:
        raise ValueError("cosines and delta norms must have identical shapes")
    if cosines.shape[0] != len(intensities) or cosines.shape[1] != len(item_ids):
        raise ValueError("cosines shape must match intensities and item_ids")

    controlled = set(int(x) for x in controlled_layers)

    samples_path = out_dir / "samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "prompt"])
        for item_id, prompt in zip(item_ids, prompts):
            w.writerow([int(item_id), str(prompt)])

    cos_path = out_dir / "cosines.csv"
    with cos_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item_id",
                "intensity",
                "layer",
                "controlled",
                "cosine",
                "delta_norm_anger",
                "delta_norm_pd",
            ]
        )
        n_layers = int(cosines.shape[2])
        for i_int, intensity in enumerate(intensities):
            for i_s, item_id in enumerate(item_ids):
                for layer in range(n_layers):
                    w.writerow(
                        [
                            int(item_id),
                            float(intensity),
                            int(layer),
                            1 if int(layer) in controlled else 0,
                            float(cosines[i_int, i_s, layer]),
                            float(delta_norms_anger[i_int, i_s, layer]),
                            float(delta_norms_pd[i_int, i_s, layer]),
                        ]
                    )


@dataclass
class AnalysisConfig:
    model_path: str
    pd_vectors_dir: Path
    split_manifest: Path
    emotion_rep_reader_path: Path
    intensities: Sequence[float]
    max_length: int
    batch_size: int
    output_root: Path


def _load_split_indices(path: Path, *, split: str) -> List[int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = str(split).strip().lower()
    if split == "test":
        key = "test_indices"
        return [int(i) for i in manifest.get(key, [])]
    if split == "train":
        key = "train_indices"
        return [int(i) for i in manifest.get(key, [])]
    if split == "all":
        train = [int(i) for i in manifest.get("train_indices", [])]
        test = [int(i) for i in manifest.get("test_indices", [])]
        return sorted(set(train + test))
    raise ValueError(f"Unknown split: {split!r} (expected test|train|all)")


def _load_pd_layer_vectors(vectors_dir: Path, control_layers: Sequence[int]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for lyr in control_layers:
        vec_path = vectors_dir / f"layer_{int(lyr)}.npy"
        if not vec_path.exists():
            raise FileNotFoundError(f"Missing PD vector for layer {lyr}: {vec_path}")
        vec = np.load(vec_path).astype(np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D PD vector at {vec_path}, got {vec.shape}")
        out[int(lyr)] = vec
    return out


def _load_emotion_layer_vectors(
    rep_reader_path: Path, *, num_layers: int, control_layers: Sequence[int], emotion: str
) -> Dict[int, np.ndarray]:
    """
    Loads the pickled emotion rep readers and extracts the requested emotion direction for each layer.
    Stored as (1, hidden), so we take [0].
    """
    raw = pickle.loads(rep_reader_path.read_bytes())
    if not isinstance(raw, dict) or str(emotion) not in raw:
        raise ValueError(f"RepReader pickle must be dict with key '{emotion}': {rep_reader_path}")
    rr = raw[str(emotion)]
    if not hasattr(rr, "directions"):
        raise ValueError(f"{emotion} RepReader missing directions: {type(rr)}")

    out: Dict[int, np.ndarray] = {}
    for lyr in control_layers:
        key = repreader_key_for_layer(layer=int(lyr), num_layers=int(num_layers))
        if key not in rr.directions:
            raise KeyError(f"Missing {emotion} direction for key {key} (layer {lyr})")
        vec = np.asarray(rr.directions[key], dtype=np.float32)
        if vec.ndim != 2 or vec.shape[0] != 1:
            raise ValueError(f"Expected {emotion} direction shape (1, hidden), got {vec.shape} for layer {lyr}")
        out[int(lyr)] = vec[0]
    return out


def _iter_batches(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sample-level per-layer cosine(Δ^emotion, Δ^pd) on Prisoner's Dilemma prompts."
    )
    parser.add_argument(
        "--model",
        default="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--pd_vectors_dir",
        default="auto_experiments/task_similarity/results/steering_vectors/Qwen2.5-3B-Instruct/20251201_131429/seed_20/layer_vectors",
    )
    parser.add_argument(
        "--split_manifest",
        default="auto_experiments/task_similarity/results/steering_vectors/Qwen2.5-3B-Instruct/20251201_131429/seed_20/split_manifest.json",
    )
    parser.add_argument(
        "--emotion_rep_reader",
        default="neuro_manipulation/representation_storage",
        help="Pickle containing emotion RepReaders (must contain key for --emotion).",
    )
    parser.add_argument("--emotion", default="anger", help="Emotion key in the RepReader pickle.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train", "all"],
        help="Which PD split to run on (default: test; use all to disable filtering).",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Top-level identifier for the run directory (default: current datetime).",
    )
    parser.add_argument(
        "--intensities",
        default="0.4,0.6,0.8,1.0,1.2,1.5",
        help="Comma-separated steering intensities.",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Passed to transformers `from_pretrained` (e.g. "auto").',
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of PD test samples (for quick sanity runs).",
    )
    parser.add_argument(
        "--output_root",
        default="auto_experiments/task_similarity/results/anger_pd_delta_similarity",
    )
    args = parser.parse_args()

    cfg = AnalysisConfig(
        model_path=str(args.model),
        pd_vectors_dir=Path(args.pd_vectors_dir),
        split_manifest=Path(args.split_manifest),
        emotion_rep_reader_path=Path(args.emotion_rep_reader),
        intensities=[float(x) for x in str(args.intensities).split(",") if str(x).strip()],
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        output_root=Path(args.output_root),
    )

    # Heavy imports kept in main to keep unit tests light.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from emotion_experiment_engine.data_models import BenchmarkConfig
    from emotion_experiment_engine.datasets.games import GameTheoryDataset
    from emotion_experiment_engine.game_prompt_wrapper import GameBenchmarkPromptWrapper
    from neuro_manipulation.prompt_formats import PromptFormat

    from .run_pd_defection_experiment import _register_control_hook

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16,
        device_map=str(args.device_map),
        trust_remote_code=True,
    )

    print(
        json.dumps(
            {
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_device_count": int(torch.cuda.device_count()),
                "requested_device_map": str(args.device_map),
                "hf_device_map_present": bool(getattr(model, "hf_device_map", None)),
                "param_device": str(next(model.parameters()).device),
                "param_dtype": str(next(model.parameters()).dtype),
            },
            indent=2,
        ),
        file=sys.stderr,
        flush=True,
    )

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Model config missing num_hidden_layers")

    control_layers = middle_third_layers(int(num_layers))
    measurement_layers = list(range(int(num_layers)))

    pd_vectors = _load_pd_layer_vectors(cfg.pd_vectors_dir, control_layers)
    emotion_vectors = _load_emotion_layer_vectors(
        cfg.emotion_rep_reader_path,
        num_layers=int(num_layers),
        control_layers=control_layers,
        emotion=str(args.emotion),
    )

    # Build PD prompts via benchmark wrapper to keep formatting consistent with the rest of repo.
    bench = BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    prompt_format = PromptFormat(tokenizer)
    game_prompt = GameBenchmarkPromptWrapper(prompt_format, bench.task_type)
    prompt_wrapper = partial(
        game_prompt.__call__,
        user_messages="Please provide your answer.",
        enable_thinking=False,
        augmentation_config=bench.augmentation_config,
        emotion=None,
    )

    dataset = GameTheoryDataset(
        config=bench,
        prompt_wrapper=prompt_wrapper,
        max_context_length=None,
        tokenizer=tokenizer,
        truncation_strategy=bench.truncation_strategy,
        answer_wrapper=None,
    )

    dataset_split = str(args.split).strip().lower()
    if dataset_split != "all":
        allowed = set(_load_split_indices(cfg.split_manifest, split=dataset_split))
        filtered_items = []
        for item in dataset.items:
            try:
                idx = int(item.id)
            except Exception:
                continue
            if idx in allowed:
                filtered_items.append(item)
        dataset.items = filtered_items

    max_samples = None if args.max_samples is None else int(args.max_samples)
    if max_samples is not None and max_samples > 0:
        dataset.items = dataset.items[:max_samples]

    prompts: List[str] = []
    item_ids: List[int] = []
    for i in range(len(dataset)):
        row = dataset[i]
        prompts.append(row["prompt"])
        item_ids.append(int(row["item"].id))

    run_id = str(args.run_id) if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    split_seed = int(json.loads(cfg.split_manifest.read_text())["split_seed"])
    out_dir = build_output_dir(
        output_root=cfg.output_root,
        run_id=run_id,
        model_path=cfg.model_path,
        emotion=str(args.emotion),
        split_seed=split_seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cosines = np.empty((len(cfg.intensities), len(prompts), len(measurement_layers)), dtype=np.float32)
    anger_norms = np.empty_like(cosines)
    pd_norms = np.empty_like(cosines)

    device = next(model.parameters()).device
    model.eval()

    def _forward_final_hidden(batch_prompts: Sequence[str]) -> np.ndarray:
        enc = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        lengths = enc["attention_mask"].sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        bs = enc["input_ids"].shape[0]
        idxs = torch.arange(bs, device=device)
        per_layer = []
        for hs in outputs.hidden_states[1:]:
            per_layer.append(hs[idxs, lengths])
        stacked = torch.stack(per_layer, dim=1)  # (bs, layers, hidden)
        return stacked.detach().cpu().float().numpy()

    batch_starts = list(range(0, len(prompts), cfg.batch_size))
    for batch_start in progress(batch_starts, total=len(batch_starts), desc="batches"):
        batch_prompts = prompts[batch_start : batch_start + cfg.batch_size]
        base = _forward_final_hidden(batch_prompts)

        for i_int, intensity in enumerate(progress(cfg.intensities, total=len(cfg.intensities), desc="intensities")):
            handles: List[Any] = []
            try:
                for lyr in control_layers:
                    layer_module = model.model.layers[int(lyr)]
                    handles.append(_register_control_hook(layer_module, emotion_vectors[int(lyr)], float(intensity)))
                steered_anger = _forward_final_hidden(batch_prompts)
            finally:
                for h in handles:
                    h.remove()

            handles = []
            try:
                for lyr in control_layers:
                    layer_module = model.model.layers[int(lyr)]
                    handles.append(_register_control_hook(layer_module, pd_vectors[int(lyr)], float(intensity)))
                steered_pd = _forward_final_hidden(batch_prompts)
            finally:
                for h in handles:
                    h.remove()

            delta_anger = steered_anger - base
            delta_pd = steered_pd - base

            cos = cosine_per_layer(delta_anger, delta_pd, eps=1e-12)
            na = np.linalg.norm(delta_anger.astype(np.float32), axis=-1).astype(np.float32)
            nb = np.linalg.norm(delta_pd.astype(np.float32), axis=-1).astype(np.float32)

            sl = slice(batch_start, batch_start + len(batch_prompts))
            cosines[i_int, sl, :] = cos
            anger_norms[i_int, sl, :] = na
            pd_norms[i_int, sl, :] = nb

    np.save(out_dir / "cosines.npy", cosines)
    np.save(out_dir / "delta_norms_anger.npy", anger_norms)
    np.save(out_dir / "delta_norms_pd.npy", pd_norms)
    write_csv_outputs(
        out_dir=out_dir,
        item_ids=item_ids,
        prompts=prompts,
        intensities=cfg.intensities,
        controlled_layers=control_layers,
        cosines=cosines,
        delta_norms_anger=anger_norms,
        delta_norms_pd=pd_norms,
    )

    meta = {
        "model_path": cfg.model_path,
        "emotion": str(args.emotion),
        "num_layers": int(num_layers),
        "controlled_layers": control_layers,
        "measurement_layers": measurement_layers,
        "intensities": list(cfg.intensities),
        "max_length": int(cfg.max_length),
        "batch_size": int(cfg.batch_size),
        "dataset_split": dataset_split,
        "split_manifest": str(cfg.split_manifest),
        "pd_vectors_dir": str(cfg.pd_vectors_dir),
        "emotion_rep_reader_path": str(cfg.emotion_rep_reader_path),
        "n_samples": len(prompts),
        "item_ids": item_ids,
        "run_id": run_id,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(str(out_dir), flush=True)


if __name__ == "__main__":
    main()
