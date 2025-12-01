"""
Responsible: auto_experiments/task_similarity/run_pd_defection_pd_behavior.py
Purpose: Generic behavior runner for contrastive activation vectors.

Given:
  - a trained model,
  - a benchmark config (e.g., game_theory / Prisoners_Dilemma),
  - a directory of per-layer steering vectors, and
  - a split manifest that records the dataset test indices,

this script:
  1) Loads the benchmark dataset.
  2) Restricts it to the recorded test split.
  3) Applies steering vectors at specified layers and intensities.
  4) Measures the fraction of a target option (e.g., "Defect") being chosen.

This keeps training (vector extraction) and behavior evaluation decoupled:
  - Training is responsible for producing vectors_dir + split_manifest.json.
  - Behavior only consumes those two artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.games import GameTheoryDataset
from emotion_experiment_engine.game_prompt_wrapper import GameBenchmarkPromptWrapper
from neuro_manipulation.prompt_formats import PromptFormat

from .run_pd_defection_experiment import _register_control_hook


@dataclass
class BehaviorConfig:
    model_path: str
    benchmark_config: Path
    vectors_dir: Path
    split_manifest: Path
    intensities: Sequence[float]
    max_length: int
    batch_size: int
    seed: int
    steering_mode: str  # "single" or "middle_third"
    single_layer: int | None


def _load_benchmark_config(path: Path) -> BenchmarkConfig:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = cfg["name"]
    task_type = cfg["task_type"]

    return BenchmarkConfig(
        name=name,
        task_type=task_type,
        data_path=None,
        base_data_dir=cfg.get("base_data_dir"),
        sample_limit=cfg.get("sample_limit"),
        augmentation_config=cfg.get("augmentation_config"),
        enable_auto_truncation=bool(cfg.get("enable_auto_truncation", False)),
        truncation_strategy=str(cfg.get("truncation_strategy", "right")),
        preserve_ratio=float(cfg.get("preserve_ratio", 1.0)),
        llm_eval_config=cfg.get("llm_eval_config"),
    )


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _compute_defect_ratio(
    model: Any,
    tokenizer: Any,
    dataset: GameTheoryDataset,
    max_length: int,
    batch_size: int,
    generation_config: Dict[str, Any],
) -> float:
    """
    Compute fraction of options where option id 2 is chosen.
    Assumes GameTheoryDataset.evaluate_response returns an option_id float.
    """
    device = _device_of(model)
    model.eval()

    defect_count = 0
    valid_count = 0

    for start in range(0, len(dataset), batch_size):
        batch_items: List[Dict[str, Any]] = []
        for idx in range(start, min(start + batch_size, len(dataset))):
            batch_items.append(dataset[idx])

        prompts = [entry["prompt"] for entry in batch_items]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                repetition_penalty=generation_config["repetition_penalty"],
            )

        responses = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for entry, prompt, resp in zip(batch_items, prompts, responses):
            # Extract option id directly; avoid LLM fallback to keep evaluation local
            options = GameTheoryDataset._extract_options_from_prompt(prompt)
            choice_id = GameTheoryDataset._extract_option_from_response(resp, options)
            if choice_id is None:
                continue

            valid_count += 1
            # For Prisoner's Dilemma, option 2 corresponds to defection;
            # for other benchmarks this "target" may be interpreted differently.
            if int(choice_id) == 2:
                defect_count += 1

    if valid_count == 0:
        return float("nan")
    return float(defect_count) / float(valid_count)


def run(cfg: BehaviorConfig) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    benchmark_cfg = _load_benchmark_config(cfg.benchmark_config)
    gen_cfg = _load_generation_config(cfg.benchmark_config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Model config missing num_hidden_layers")

    if cfg.steering_mode == "middle_third":
        start = num_layers // 3
        end = (2 * num_layers) // 3
        control_layers = list(range(start, end))
    elif cfg.steering_mode == "single":
        if cfg.single_layer is None:
            raise ValueError("single_layer must be provided when steering_mode='single'")
        control_layers = [int(cfg.single_layer)]
    else:
        raise ValueError(f"Unsupported steering_mode: {cfg.steering_mode}")

    prompt_format = PromptFormat(tokenizer)
    game_prompt = GameBenchmarkPromptWrapper(prompt_format, benchmark_cfg.task_type)

    prompt_wrapper = partial(
        game_prompt.__call__,
        user_messages="Please provide your answer.",
        enable_thinking=False,
        augmentation_config=benchmark_cfg.augmentation_config,
        emotion=None,
    )

    dataset = GameTheoryDataset(
        config=benchmark_cfg,
        prompt_wrapper=prompt_wrapper,
        max_context_length=None,
        tokenizer=tokenizer,
        truncation_strategy=benchmark_cfg.truncation_strategy,
        answer_wrapper=None,
    )

    # Restrict dataset to test indices recorded in split_manifest
    manifest = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
    test_indices = set(int(i) for i in manifest.get("test_indices", []))

    filtered_items = []
    for item in dataset.items:
        try:
            idx = int(item.id)
        except Exception:
            continue
        if idx in test_indices:
            filtered_items.append(item)
    dataset.items = filtered_items
    split_info = {
        "n_test_indices": len(test_indices),
        "n_dataset_items": len(filtered_items),
    }

    # Baseline (no steering)
    defect_ratio: Dict[float, float] = {}
    intensities_list = [float(x) for x in cfg.intensities]
    if 0.0 not in intensities_list:
        intensities_list.insert(0, 0.0)

    base_ratio = _compute_defect_ratio(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        generation_config=gen_cfg,
    )
    defect_ratio[0.0] = base_ratio

    # Steered runs
    if cfg.steering_mode == "middle_third":
        layer_vectors: Dict[int, np.ndarray] = {}
        for lyr in control_layers:
            vec_path = cfg.vectors_dir / f"layer_{lyr}.npy"
            layer_vectors[lyr] = np.load(vec_path)
    else:
        vec_path = cfg.vectors_dir / f"layer_{control_layers[0]}.npy"
        base_vec = np.load(vec_path)

    for inten in intensities_list:
        if float(inten) == 0.0:
            continue
        handles: List[Any] = []
        try:
            for lyr in control_layers:
                try:
                    layer_module = model.model.layers[lyr]
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(f"Cannot locate layer {lyr} on model") from exc

                if cfg.steering_mode == "middle_third":
                    vec = layer_vectors[lyr]
                else:
                    vec = base_vec

                handles.append(_register_control_hook(layer_module, vec, float(inten)))

            ratio = _compute_defect_ratio(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
                generation_config=gen_cfg,
            )
        finally:
            for h in handles:
                h.remove()
        defect_ratio[float(inten)] = ratio

    cfg_output_dir = Path(cfg_output_dir_from_vectors(cfg.vectors_dir, cfg.model_path, output_dir=None))
    cfg_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cfg_output_dir / f"{Path(cfg.model_path).name}_behavior_{timestamp}.json"

    result = {
        "model_path": cfg.model_path,
        "benchmark_config": str(cfg.benchmark_config),
        "vectors_dir": str(cfg.vectors_dir),
        "split_manifest": str(cfg.split_manifest),
        "timestamp": timestamp,
        "seed": cfg.seed,
        "benchmark_name": benchmark_cfg.name,
        "task_type": benchmark_cfg.task_type,
        "steering_mode": cfg.steering_mode,
        "control_layers": control_layers,
        "intensities": [float(x) for x in intensities_list],
        "defect_ratio": defect_ratio,
        "n_items": len(dataset),
        "split_info": split_info,
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _load_generation_config(cfg_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    gen = cfg.get("generation_config", {}) or {}
    return {
        "max_new_tokens": int(gen.get("max_new_tokens", 256)),
        "temperature": float(gen.get("temperature", 0.0)),
        "top_p": float(gen.get("top_p", 1.0)),
        "do_sample": bool(gen.get("do_sample", False)),
        "repetition_penalty": float(gen.get("repetition_penalty", 1.0)),
    }


def cfg_output_dir_from_vectors(vectors_dir: Path, model_path: str, output_dir: Path | None) -> str:
    if output_dir is not None:
        return str(output_dir)
    # Default: sibling directory "behavior" next to vectors_dir
    return str(vectors_dir.parent / "behavior")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark_config", required=True)
    parser.add_argument("--vectors_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--intensities", type=str, default="0.0,0.5,1.0,1.5,2.0")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steering_mode",
        type=str,
        default="single",
        choices=["single", "middle_third"],
        help="Steering scheme: 'single' layer or 'middle_third' of layers.",
    )
    parser.add_argument(
        "--single_layer",
        type=int,
        default=None,
        help="Layer index to steer when steering_mode='single'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional explicit output directory for behavior results.",
    )
    args = parser.parse_args()

    intensity_vals = [float(x) for x in args.intensities.split(",") if x]

    cfg = BehaviorConfig(
        model_path=args.model,
        benchmark_config=Path(args.benchmark_config),
        vectors_dir=Path(args.vectors_dir),
        split_manifest=Path(args.split_manifest),
        intensities=intensity_vals,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        steering_mode=args.steering_mode,
        single_layer=args.single_layer,
    )

    # Compute output dir (may be derived from vectors_dir if not provided)
    global cfg_output_dir  # simple closure pass-through
    cfg_output_dir = Path(args.output_dir) if args.output_dir is not None else None

    result = run(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

