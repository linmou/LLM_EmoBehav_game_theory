"""
Responsible: auto_experiments/task-similarity/run_pd_defection_experiment.py
Purpose: Train defection activation vectors on PD data, validate per layer, and evaluate behavior shift.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_experiments.task_similarity.pd_data import (
    build_pd_pair_bundle,
    build_repreader_dataset,
)
from auto_experiments.task_similarity.pd_prompt_builder import (
    PromptPair,
    build_inference_prompt,
)
from auto_experiments.task_similarity.pd_vector_extractor import (
    compute_vectors_and_accuracy,
    select_best_layer,
)
import torch.nn as nn


def _collect_hidden(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    batch_size: int = 8,
    max_length: int = 256,
) -> Dict[int, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    out: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
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
            outputs = model(**enc, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # len = num_layers + 1
        mask = enc["attention_mask"].unsqueeze(-1)
        mask_sum = mask.sum(dim=1).clamp(min=1)
        for layer in layers:
            hs = hidden_states[layer + 1]
            masked_mean = (hs * mask).sum(dim=1) / mask_sum
            out[layer].append(masked_mean.detach().cpu().float())
    return {k: torch.cat(v, dim=0).numpy() for k, v in out.items()}


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
            other_token = label_to_token["A" if label == "B" else "B"]
            if (defect_token == label_to_token["A"] and a_scores[idx] > b_scores[idx]) or (
                defect_token == label_to_token["B"] and b_scores[idx] > a_scores[idx]
            ):
                wins += 1
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


def run(
    model_path: str,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 8,
    seed: int = 0,
    intensity: float = 1.0,
    max_pairs: int | None = None,
    middle_third_only: bool = False,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    bundle = build_pd_pair_bundle(
        Path("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json"),
        seed=seed,
    )
    train_pairs = bundle.train_pairs
    test_pairs = bundle.test_pairs
    if max_pairs is not None:
        train_pairs = train_pairs[:max_pairs]
        test_pairs = test_pairs[:max_pairs]

    train_ds = build_repreader_dataset(train_pairs)
    test_ds = build_repreader_dataset(test_pairs)

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
    if middle_third_only:
        start = num_layers // 3
        end = (2 * num_layers) // 3
        control_layers = list(range(start, end))
    else:
        control_layers = list(range(num_layers))

    train_hidden = _collect_hidden(model, tokenizer, train_ds["data"], control_layers, batch_size, max_length)
    test_hidden = _collect_hidden(model, tokenizer, test_ds["data"], control_layers, batch_size, max_length)

    layer_results = compute_vectors_and_accuracy(train_hidden, test_hidden)
    best_layer, best = select_best_layer(layer_results)

    label_to_token = {"A": _token_id(tokenizer, "A"), "B": _token_id(tokenizer, "B")}

    def _apply_hooks():
        handles = []
        if middle_third_only:
            for layer_id in control_layers:
                vec = layer_results[layer_id].vector
                target_layer = model.model.layers[layer_id]
                handles.append(_register_control_hook(target_layer, vec, intensity))
        else:
            target_layer = model.model.layers[best_layer]
            handles.append(_register_control_hook(target_layer, best.vector, intensity))
        return handles

    base_rate = _decision_rate(model, tokenizer, test_pairs, label_to_token, batch_size, max_length)
    handles = _apply_hooks()
    try:
        steered_rate = _decision_rate(model, tokenizer, test_pairs, label_to_token, batch_size, max_length)
    finally:
        for h in handles:
            h.remove()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{Path(model_path).name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model_path": model_path,
        "timestamp": timestamp,
        "seed": seed,
        "max_pairs": max_pairs,
        "control_layers": control_layers,
        "best_layer": best_layer,
        "best_accuracy": best.accuracy,
        "layer_accuracies": {k: v.accuracy for k, v in layer_results.items()},
        "base_defect_rate": base_rate,
        "steered_defect_rate": steered_rate,
        "intensity": intensity,
        "middle_third_only": middle_third_only,
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.save(run_dir / "best_vector.npy", best.vector)
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
    args = parser.parse_args()

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
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
