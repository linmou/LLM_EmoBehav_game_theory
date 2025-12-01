"""
Responsible: auto_experiments/task-similarity/compute_pd_delta.py
Purpose: Compute delta activations for PD defection vectors using delta_activation_engine probes.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "none")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from delta_activation_engine.prompts.probes_texts import get_generic_probes
from delta_activation_engine.backends.hf import select_middle_third_layers


def compute_delta(baseline: np.ndarray, steered: np.ndarray) -> np.ndarray:
    if baseline.shape != steered.shape:
        raise ValueError(f"shape mismatch: baseline {baseline.shape} vs steered {steered.shape}")
    return (steered - baseline).astype(np.float32)


def _load_vectors(vector_path: Path, control_layers: Sequence[int]) -> Dict[int, np.ndarray]:
    if vector_path.is_dir():
        vectors: Dict[int, np.ndarray] = {}
        for layer_idx in control_layers:
            path = vector_path / f"layer_{layer_idx}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing vector file for layer {layer_idx}: {path}")
            vectors[layer_idx] = np.load(path)
        return vectors
    vec = np.load(vector_path)
    return {int(layer_idx): vec for layer_idx in control_layers}


def _collect_hidden(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    batch_size: int = 8,
    max_length: int = 256,
) -> Dict[int, np.ndarray]:
    device = next(model.parameters()).device
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
        mask = enc["attention_mask"].unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        for layer in layers:
            hs = outputs.hidden_states[layer + 1]  # layer 0 is embedding
            pooled = (hs * mask).sum(dim=1) / denom
            out[layer].append(pooled.detach().cpu().float())
    return {k: torch.cat(v, dim=0).mean(dim=0).numpy() for k, v in out.items()}


def _collect_final_token_hidden(
    model,
    tokenizer,
    prompts: Sequence[str],
    measurement_layer: int,
    batch_size: int = 8,
    max_length: int = 256,
) -> np.ndarray:
    device = next(model.parameters()).device
    outs: List[torch.Tensor] = []
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
        hs = outputs.hidden_states[measurement_layer + 1]
        lengths = enc["attention_mask"].sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        final = hs[torch.arange(hs.size(0), device=hs.device), lengths]
        outs.append(final.detach().cpu().float())
    return torch.cat(outs, dim=0).mean(dim=0).numpy()


def _register_control_hook(layer_module, vec: np.ndarray, intensity: float):
    vec_t = torch.tensor(vec * intensity, device=next(layer_module.parameters()).device)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            ctrl = vec_t.to(hidden.dtype).view(1, 1, -1)
            hidden = hidden + ctrl
            return (hidden,) + output[1:]
        ctrl = vec_t.to(output.dtype).view(1, 1, -1)
        return output + ctrl

    return layer_module.register_forward_hook(hook)


def resolve_control_layers(num_layers: int, layer: Union[int, None], use_middle_third: bool) -> List[int]:
    if use_middle_third:
        return select_middle_third_layers(num_layers)
    if layer is None:
        raise ValueError("Specify layer or enable use_middle_third")
    return [int(layer)]


def run_delta(
    model_path: str,
    vector_path: Path,
    layer: Union[int, None],
    use_middle_third: bool,
    intensity: float,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 8,
    seed: int = 0,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    if not use_middle_third and layer is None:
        raise ValueError("Specify layer or enable use_middle_third")

    prompts = get_generic_probes()

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
    control_layers = resolve_control_layers(num_layers, layer, use_middle_third)

    measurement_layer = num_layers - 1
    baseline_vec = _collect_final_token_hidden(
        model, tokenizer, prompts, measurement_layer, batch_size=batch_size, max_length=max_length
    )

    vector_map = _load_vectors(vector_path, control_layers)
    handles = []
    try:
        for lyr in control_layers:
            try:
                target_layer = model.model.layers[lyr]
            except Exception as exc:
                raise RuntimeError(f"Cannot locate layer {lyr} on model") from exc
            vec = vector_map[lyr]
            handles.append(_register_control_hook(target_layer, vec, intensity))
        steered_vec = _collect_final_token_hidden(
            model, tokenizer, prompts, measurement_layer, batch_size=batch_size, max_length=max_length
        )
    finally:
        for h in handles:
            h.remove()

    baseline = {measurement_layer: baseline_vec}
    steered = {measurement_layer: steered_vec}
    delta = {measurement_layer: compute_delta(baseline_vec, steered_vec)}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{Path(model_path).name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "baseline.npz", **{str(k): v for k, v in baseline.items()})
    np.savez(run_dir / "steered.npz", **{str(k): v for k, v in steered.items()})
    np.savez(run_dir / "delta.npz", **{str(k): v for k, v in delta.items()})

    result = {
        "model_path": model_path,
        "vector_path": str(vector_path),
        "control_layers": control_layers,
        "measurement_layer": measurement_layer,
        "intensity": intensity,
        "seed": seed,
        "timestamp": timestamp,
        "prompt_hash": hash(tuple(prompts)),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector_path", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Target layer. If omitted with --middle_third, use middle third.")
    parser.add_argument("--middle_third", action="store_true", help="Apply vector to middle third of layers.")
    parser.add_argument("--intensity", type=float, default=1.5)
    parser.add_argument("--output_dir", default="auto_experiments/task-similarity/results/delta")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.middle_third and args.layer is None:
        raise ValueError("Specify --layer or use --middle_third")
    result = run_delta(
        model_path=args.model,
        vector_path=Path(args.vector_path),
        layer=args.layer,
        use_middle_third=args.middle_third,
        intensity=args.intensity,
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
