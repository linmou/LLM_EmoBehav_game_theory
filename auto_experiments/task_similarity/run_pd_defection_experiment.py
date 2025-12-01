"""
Responsible: auto_experiments/task-similarity/run_pd_defection_experiment.py
Purpose: Train contrastive defection activation vectors on Prisoner's Dilemma
         data, validate per layer, and evaluate behavior shift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from .pd_data import build_pd_pair_bundle, build_repreader_dataset
from .pd_prompt_builder import PromptPair, build_inference_prompt
from .pd_hidden_extractor import collect_answer_means

import logging

logger = logging.getLogger(__name__)


def _token_id(tokenizer, token_str: str) -> int:
    ids = tokenizer(token_str, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"Token '{token_str}' splits into {ids}")
    return ids[0]


def _decision_rate(
    model,
    tokenizer,
    pairs: Sequence[PromptPair],
    label_to_token: Dict[str, int],
    batch_size: int = 8,
    max_length: int = 256,
) -> float:
    device = next(model.parameters()).device
    model.eval()
    prompts = [
        build_inference_prompt(p.meta.description, p.meta.opt_a, p.meta.opt_b) for p in pairs
    ]
    labels = [p.meta.defect_label for p in pairs]
    wins = 0
    total = 0
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        enc = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        last_logits = logits[:, -1, :]
        a_scores = last_logits[:, label_to_token["A"]]
        b_scores = last_logits[:, label_to_token["B"]]
        for idx, label in enumerate(batch_labels):
            defect_token = label_to_token[label]
            if defect_token == label_to_token["A"]:
                wins += float(a_scores[idx] > b_scores[idx])
            else:
                wins += float(b_scores[idx] > a_scores[idx])
            total += 1
    return wins / total if total else 0.0


def _register_control_hook(
    layer: nn.Module, vec: np.ndarray, intensity: float
):
    vec_t = torch.tensor(vec * intensity, device=next(layer.parameters()).device)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            ctrl = vec_t.to(hidden.dtype).view(1, 1, -1)
            hidden = hidden + ctrl
            return (hidden,) + output[1:]
        ctrl = vec_t.to(output.dtype).view(1, 1, -1)
        return output + ctrl

    return layer.register_forward_hook(hook)


def train_pd_repreader(
    model: Any,
    tokenizer: Any,
    train_data: Dict[str, Any],
    test_data: Dict[str, Any],
    hidden_layers: Sequence[int],
    batch_size: int,
    max_length: int,
    span_mode: str = "assistant",
) -> Tuple[Any, Dict[int, float], Dict[int, np.ndarray]]:
    """
    Train a PCA-based defection direction for PD using assistant-span mean
    hidden states as the representation.

    Returns:
        rep_reader: currently unused (kept for API compatibility; set to None)
        layer_acc: per-layer validation accuracy on test_data
        layer_vectors: oriented direction vector per layer (1D np.ndarray)
    """
    # Extract span-mean representations for train and test data
    hidden_layers_list = list(hidden_layers)
    train_hiddens = collect_answer_means(
        model=model,
        tokenizer=tokenizer,
        prompts=train_data["data"],
        layers=hidden_layers_list,
        max_length=max_length,
        batch_size=batch_size,
        span=span_mode,
    )
    test_hiddens = collect_answer_means(
        model=model,
        tokenizer=tokenizer,
        prompts=test_data["data"],
        layers=hidden_layers_list,
        max_length=max_length,
        batch_size=batch_size,
        span=span_mode,
    )

    # Sanity checks: number of samples must match labels length (flattened)
    total_train_examples = len(train_data["data"])
    total_label_slots = len(np.concatenate(train_data["labels"]))
    assert (
        total_train_examples == total_label_slots
    ), f"Train data/labels mismatch: {total_train_examples} examples vs {total_label_slots} label slots"
    for layer in hidden_layers_list:
        assert (
            train_hiddens[layer].shape[0] == total_train_examples
        ), f"Train hidden count mismatch at layer {layer}"
        assert (
            test_hiddens[layer].shape[0] == len(test_data["data"])
        ), f"Test hidden count mismatch at layer {layer}"

    layer_acc: Dict[int, float] = {}
    layer_vectors: Dict[int, np.ndarray] = {}

    for layer in tqdm(
        hidden_layers_list,
        desc="Training PD defection directions",
        leave=False,
    ):
        H_train = train_hiddens[layer]  # (N_train, hidden)
        H_test = test_hiddens[layer]    # (N_test, hidden)

        # We expect paired ordering: [pos0, neg0, pos1, neg1, ...]
        assert H_train.shape[0] % 2 == 0, f"Train examples for layer {layer} not even; cannot form pairs"
        assert H_test.shape[0] % 2 == 0, f"Test examples for layer {layer} not even; cannot form pairs"

        # Build difference matrix for PCA: pos - neg per pair
        pos_train = H_train[0::2]
        neg_train = H_train[1::2]
        diffs = pos_train - neg_train  # (num_pairs, hidden)
        diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)

        # If we have no pairs, fall back to zero vector and zero accuracy
        if diffs_centered.shape[0] == 0:
            hidden_dim = H_train.shape[1]
            layer_vectors[layer] = np.zeros(hidden_dim, dtype=np.float32)
            layer_acc[layer] = 0.0
            continue

        # PCA via SVD on the centered differences
        # diffs_centered: (num_pairs, hidden_dim)
        U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
        direction = Vt[0].astype(np.float32)  # principal component, shape (hidden_dim,)

        # Compute test projections
        scores = H_test @ direction  # (N_test,)
        # Test examples must be arranged as [pos0, neg0, pos1, neg1, ...]
        assert scores.shape[0] == len(test_data["data"])
        pairs: List[Tuple[float, float]] = []
        for i in range(0, len(scores), 2):
            if i + 1 < len(scores):
                pairs.append((scores[i], scores[i + 1]))

        # Decide orientation: choose sign that maximizes correctness on held-out pairs
        pos_scores = np.array([p[0] for p in pairs], dtype=np.float32)
        neg_scores = np.array([p[1] for p in pairs], dtype=np.float32)
        acc_plus = float((pos_scores > neg_scores).mean()) if len(pairs) else 0.0
        acc_minus = float((pos_scores < neg_scores).mean()) if len(pairs) else 0.0

        if acc_minus > acc_plus:
            sign = -1.0
            acc = acc_minus
        else:
            sign = 1.0
            acc = acc_plus

        layer_acc[layer] = acc

        layer_vectors[layer] = direction * sign

    # rep_reader is unused by downstream code; keep API stable by returning None
    return None, layer_acc, layer_vectors


def run(
    model_path: str,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 8,
    seed: int = 0,
    intensity: float = 1.0,
    max_pairs: int | None = None,
    middle_third_only: bool = False,
    behavior_intensities: Sequence[float] | None = None,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_path = Path(
        "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json"
    )
    logger.info("Loading contrastive dataset from %s", dataset_path)
    bundle = build_pd_pair_bundle(dataset_path, seed=seed)
    train_pairs = bundle.train_pairs
    test_pairs = bundle.test_pairs
    if max_pairs is not None:
        train_pairs = train_pairs[:max_pairs]
        test_pairs = test_pairs[:max_pairs]

    train_ds = build_repreader_dataset(train_pairs)
    test_ds = build_repreader_dataset(test_pairs)

    logger.info(
        "Prepared %d total pairs (%d train / %d test)",
        len(bundle.pairs),
        len(train_pairs),
        len(test_pairs),
    )

    logger.info("Loading model %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Model config missing num_hidden_layers")
    logger.info("Model has %d transformer layers", num_layers)
    if middle_third_only:
        start = num_layers // 3
        end = (2 * num_layers) // 3
        control_layers = list(range(start, end))
    else:
        control_layers = list(range(num_layers))
    # Train PD defection directions and compute per-layer accuracy
    logger.info(
        "Training contrastive directions on %d train prompts (span_mode=option)",
        len(train_ds["data"]),
    )
    rep_reader, layer_acc, layer_vectors = train_pd_repreader(
        model=model,
        tokenizer=tokenizer,
        train_data=train_ds,
        test_data=test_ds,
        hidden_layers=control_layers,
        batch_size=batch_size,
        max_length=max_length,
        span_mode="option",
    )

    # Select best layer by validation accuracy
    best_layer = max(layer_acc.items(), key=lambda kv: kv[1])[0]
    best_accuracy = layer_acc[best_layer]

    label_to_token = {"A": _token_id(tokenizer, "A"), "B": _token_id(tokenizer, "B")}

    logger.info("Measuring baseline defection rate on held-out set")
    base_rate = _decision_rate(
        model, tokenizer, test_pairs, label_to_token, batch_size, max_length
    )

    # Persist split manifest and per-layer vectors under a model-specific root.
    model_root = output_dir / Path(model_path).name / datetime.now().strftime("%Y%m%d_%H%M%S") / f"seed_{seed}"
    model_root.mkdir(parents=True, exist_ok=True)

    # Reconstruct train/test indices relative to original dataset entries
    # so downstream behavior code can reuse the exact split.
    idx_map: Dict[int, int] = {id(p): i for i, p in enumerate(bundle.pairs)}
    train_indices: List[int] = [idx_map[id(p)] for p in train_pairs]
    test_indices: List[int] = [idx_map[id(p)] for p in test_pairs]

    # Compute dataset hash and per-entry hashes for integrity checks
    raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    entry_hashes: Dict[str, str] = {}
    for idx in sorted(set(train_indices + test_indices)):
        entry_json = json.dumps(raw_data[idx], sort_keys=True)
        entry_hashes[str(idx)] = hashlib.sha256(entry_json.encode("utf-8")).hexdigest()

    split_manifest = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "split_seed": seed,
        "train_ratio": 0.5,
        "max_pairs": max_pairs,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "entry_hashes": entry_hashes,
    }
    with open(model_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    # Save per-layer vectors under model_root/layer_vectors
    vectors_dir = model_root / "layer_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, vec in layer_vectors.items():
        np.save(vectors_dir / f"layer_{layer_idx}.npy", vec)
    with open(output_dir / "layer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "layer_accuracies": {int(k): float(v) for k, v in layer_acc.items()},
                "best_layer": int(best_layer),
                "best_accuracy": float(best_accuracy),
            },
            f,
            indent=2,
        )

    # Behavior evaluation per layer
    intensities = list(behavior_intensities or [0.5, 1.0, 1.5, 2.0])
    per_layer_behavior: Dict[int, Dict[float, float]] = {}
    logger.info(
        "Evaluating behavior shift for %d layers at intensities=%s",
        len(layer_vectors),
        intensities,
    )
    for layer_idx, vec in tqdm(
        layer_vectors.items(),
        desc="Behavior evaluation per layer",
        leave=False,
    ):
        layer_module = model.model.layers[layer_idx]
        per_intensity: Dict[float, float] = {}
        for inten in intensities:
            handle = _register_control_hook(layer_module, vec, inten)
            try:
                rate = _decision_rate(
                    model, tokenizer, test_pairs, label_to_token, batch_size, max_length
                )
            finally:
                handle.remove()
            per_intensity[float(inten)] = rate
        per_layer_behavior[layer_idx] = per_intensity

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{Path(model_path).name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model_path": model_path,
        "timestamp": timestamp,
        "seed": seed,
        "max_pairs": max_pairs,
        "control_layers": control_layers,
        "best_layer": int(best_layer),
        "best_accuracy": float(best_accuracy),
        "layer_accuracies": {int(k): float(v) for k, v in layer_acc.items()},
        "base_defect_rate": base_rate,
        "steered_defect_rate": per_layer_behavior.get(best_layer, {}).get(intensity, base_rate),
        "intensity": intensity,
        "middle_third_only": middle_third_only,
        "behavior_defect_rates": per_layer_behavior,
        "intensities_tested": [float(x) for x in intensities],
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.save(run_dir / "best_vector.npy", layer_vectors[best_layer])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default="auto_experiments/task-similarity/results")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--middle_third_only", action="store_true")
    parser.add_argument("--behavior_intensities", type=str, default="0.5,1.0,1.5,2.0")
    args = parser.parse_args()

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )

    behavior_intensities = [float(x) for x in args.behavior_intensities.split(",") if x]
    os.makedirs(args.output_dir, exist_ok=True)
    result = run(
        model_path=args.model,
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        intensity=args.intensity,
        max_pairs=args.max_pairs,
        middle_third_only=args.middle_third_only,
        behavior_intensities=behavior_intensities,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
