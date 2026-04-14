#!/usr/bin/env python3
# Purpose: run a full self-report logprob sweep over all hidden layers while reusing one loaded vLLM model and one shared neutral baseline.

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


MODEL_PATH = "/home/jjl7137/huggingface_models/Qwen/Qwen3-0.6B"
SELF_REPORT_DATA = "data/emotion_scales/emotion_check_self_report_emotion_options6.jsonl"
STEER_EMOTIONS = ["anger", "happiness", "sadness", "fear", "disgust", "surprise"]
TARGET_EMOTIONS = STEER_EMOTIONS + ["neutral"]
SELF_REPORT_INTENSITIES = [1.0, 2.0, 4.0, 6.0, 8.0]
LOGPROBS_K = 20
PROMPT_LOGPROBS_K = 20
PREVIOUS_PD_LAYERS = {
    "anger": 17,
    "disgust": 21,
    "fear": 23,
    "happiness": 17,
    "sadness": 15,
    "surprise": 15,
}
DEFAULT_STIMULUS_DATA_DIR = "data/stimulus/crowd-enVent_textlike"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "auto_experiments" / "pd_selfreport_pd_coupling_multimodel"
DEFAULT_LEXICAL_TARGETS = {
    "anger": ["anger"],
    "happiness": ["happiness"],
    "sadness": ["sadness"],
    "fear": ["fear"],
    "disgust": ["disgust"],
    "surprise": ["surprise"],
    "neutral": ["neutral"],
}


def model_slug(model_path: str) -> str:
    return Path(str(model_path).rstrip("/")).name.lower().replace(".", "p")


def default_output_root_for_model(model_path: str, *, results_root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    slug = model_slug(model_path)
    if slug == "qwen2p5-0p5b-instruct":
        return Path(results_root) / "self_report_logprob"
    return Path(results_root) / "self_report_logprob_multimodel" / slug


OUTPUT_ROOT = default_output_root_for_model(MODEL_PATH)


@lru_cache(maxsize=1)
def _runtime() -> dict[str, Any]:
    from transformers import AutoConfig

    from auto_experiments.emotion_lexical_logprob_change.run_emotion_lexical_logprob_change import (
        _build_benchmark_config,
        _build_control_activations,
        _build_loading_config,
        _compute_prompt_logprob_results,
        _flatten_targets,
        _load_prompt_items,
        _resolve_benchmark_user_messages,
    )
    from emotion_experiment_engine.emotion_lexical_logprob import (
        build_delta_matrix,
        build_target_option_softmax_metrics,
        summarize_delta_by_steer,
        summarize_directional_match_by_steer,
        summarize_emotion_gap_by_steer,
        summarize_target_option_softmax_by_steer,
    )
    from neuro_manipulation.configs.experiment_config import get_repe_eng_config
    from neuro_manipulation.model_layer_detector import ModelLayerDetector
    from neuro_manipulation.model_utils import load_emotion_readers, setup_model_and_tokenizer
    from neuro_manipulation.repe.pipelines import repe_pipeline_registry
    from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook

    return {
        "AutoConfig": AutoConfig,
        "_build_benchmark_config": _build_benchmark_config,
        "_build_control_activations": _build_control_activations,
        "_build_loading_config": _build_loading_config,
        "_compute_prompt_logprob_results": _compute_prompt_logprob_results,
        "_flatten_targets": _flatten_targets,
        "_load_prompt_items": _load_prompt_items,
        "_resolve_benchmark_user_messages": _resolve_benchmark_user_messages,
        "build_delta_matrix": build_delta_matrix,
        "build_target_option_softmax_metrics": build_target_option_softmax_metrics,
        "summarize_delta_by_steer": summarize_delta_by_steer,
        "summarize_directional_match_by_steer": summarize_directional_match_by_steer,
        "summarize_emotion_gap_by_steer": summarize_emotion_gap_by_steer,
        "summarize_target_option_softmax_by_steer": summarize_target_option_softmax_by_steer,
        "get_repe_eng_config": get_repe_eng_config,
        "ModelLayerDetector": ModelLayerDetector,
        "load_emotion_readers": load_emotion_readers,
        "setup_model_and_tokenizer": setup_model_and_tokenizer,
        "repe_pipeline_registry": repe_pipeline_registry,
        "RepControlVLLMHook": RepControlVLLMHook,
    }


def parse_csv_arg(raw: str | None, *, cast: Callable[[str], Any]) -> list[Any] | None:
    if raw is None:
        return None
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def resolve_output_root(model_path: str, cli_output_root: Path | None) -> Path:
    if cli_output_root is not None:
        return cli_output_root
    if str(model_path) == MODEL_PATH:
        return OUTPUT_ROOT
    return default_output_root_for_model(str(model_path))


def _iter_progress(items: Iterable[Any], *, total: int, desc: str) -> Iterable[Any]:
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc, dynamic_ncols=True)


def _control_layer_from_1based(layer_1based: int, num_hidden_layers: int) -> int:
    return int(layer_1based) - (int(num_hidden_layers) + 1)


def _load_num_hidden_layers(model_path: str) -> int:
    config = _runtime()["AutoConfig"].from_pretrained(model_path, trust_remote_code=True)
    value = getattr(config, "num_hidden_layers", None)
    if value is None:
        raise ValueError(f"Could not infer num_hidden_layers from {model_path}")
    return int(value)


def build_condition_grid(
    *,
    num_hidden_layers: int,
    emotions: Sequence[str] | None = None,
    intensities: Sequence[float] | None = None,
    layers: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    use_emotions = list(STEER_EMOTIONS if emotions is None else emotions)
    use_intensities = [float(v) for v in (SELF_REPORT_INTENSITIES if intensities is None else intensities)]
    use_layers = (
        list(range(1, int(num_hidden_layers) + 1))
        if layers is None
        else [int(v) for v in layers]
    )
    grid: list[dict[str, Any]] = []
    for emotion in use_emotions:
        for layer_1based in use_layers:
            control_layer = _control_layer_from_1based(layer_1based, num_hidden_layers)
            for intensity in use_intensities:
                grid.append(
                    {
                        "emotion": emotion,
                        "layer_1based": layer_1based,
                        "control_layer": control_layer,
                        "intensity": float(intensity),
                    }
                )
    return grid


def build_run_tag(condition: dict[str, Any]) -> str:
    intensity = f"{float(condition['intensity']):.1f}"
    return (
        f"{condition['emotion']}_layer_{int(condition['layer_1based'])}_"
        f"intensity_{intensity.replace('.', 'p')}"
    )


def condition_output_dir(output_root: Path, condition: dict[str, Any]) -> Path:
    return output_root / build_run_tag(condition)


def select_shard(
    *,
    grid: Sequence[dict[str, Any]],
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    if int(num_shards) <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError(f"shard_index must be in [0, {int(num_shards) - 1}], got {shard_index}")
    return [condition for idx, condition in enumerate(grid) if idx % int(num_shards) == int(shard_index)]


def _base_config(
    model_path: str,
    *,
    tensor_parallel_size: int = 1,
    stimulus_data_dir: str = DEFAULT_STIMULUS_DATA_DIR,
) -> dict[str, Any]:
    return {
        "model": model_path,
        "steer_emotions": STEER_EMOTIONS,
        "target_emotions": TARGET_EMOTIONS,
        "lexical_targets": DEFAULT_LEXICAL_TARGETS,
        "token_pos": "end",
        "benchmark": {
            "name": "emotion_check",
            "task_type": "self_report_emotion_options6",
            "data_path": SELF_REPORT_DATA,
            "base_data_dir": "data/emotion_scales",
            "sample_limit": None,
            "enable_auto_truncation": False,
            "option_shuffle_method": "per_item",
            "option_shuffle_seed": 42,
        },
        "repe_eng_config": {
            "direction_method": "pca",
            "data_dir": str(stimulus_data_dir),
        },
        "loading_config": {
            "model_path": model_path,
            "gpu_memory_utilization": 0.45,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": 4096,
            "enforce_eager": True,
            "quantization": None,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "seed": 42,
            "disable_custom_all_reduce": False,
            "additional_vllm_kwargs": {},
        },
        "generation_config": {
            "enable_thinking": False,
        },
        "output_config": {
            "save_full_condition_artifacts": False,
        },
    }


def _compute_rows_for_condition(
    *,
    rep_control_hook: Any,
    tokenizer: Any,
    prompt_items: Sequence[dict[str, Any]],
    target_pairs: Sequence[tuple[str, str]],
    steer_emotion: str,
    steer_intensity: float,
    activations: dict[int, Any] | None,
    chunk_size: int,
) -> list[dict[str, Any]]:
    runtime = _runtime()
    rows: list[dict[str, Any]] = []
    requests = [
        {
            "item_id": item["item_id"],
            "input_text": item["input_text"],
            "prompt": item["prompt"],
            "target_emotion": target_emotion,
            "target_sequence": target_sequence,
        }
        for item in prompt_items
        for target_emotion, target_sequence in target_pairs
    ]
    for start in _iter_progress(
        range(0, len(requests), chunk_size),
        total=(len(requests) + chunk_size - 1) // chunk_size,
        desc=f"{steer_emotion}@{steer_intensity:g}",
    ):
        batch = requests[start : start + chunk_size]
        outputs = rep_control_hook(
            text_inputs=[row["prompt"] + row["target_sequence"] for row in batch],
            activations=activations,
            token_pos="end",
            max_new_tokens=1,
            temperature=0.0,
            logprobs=LOGPROBS_K,
            prompt_logprobs=PROMPT_LOGPROBS_K,
        )
        for row, output in zip(batch, outputs):
            prob_result = runtime["_compute_prompt_logprob_results"](
                outputs=[output],
                tokenizer=tokenizer,
                prompt=row["prompt"],
                target_sequences=[row["target_sequence"]],
            )
            result = None if not prob_result else prob_result[0]
            rows.append(
                {
                    "item_id": row["item_id"],
                    "input_text": row["input_text"],
                    "prompt": row["prompt"],
                    "steer_emotion": steer_emotion,
                    "steer_intensity": float(steer_intensity),
                    "target_emotion": row["target_emotion"],
                    "target_sequence": row["target_sequence"],
                    "log_prob": None if result is None else result["log_prob"],
                    "prob": None if result is None else result["prob"],
                    "num_tokens": None if result is None else result["num_tokens"],
                    "available": result is not None,
                }
            )
    return rows


def _write_condition_outputs(
    *,
    model_path: str,
    stimulus_data_dir: str,
    output_dir: Path,
    baseline_rows: Sequence[dict[str, Any]],
    steered_rows: Sequence[dict[str, Any]],
    condition: dict[str, Any],
    save_full_condition_artifacts: bool,
) -> None:
    runtime = _runtime()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / "raw_sequence_logprobs.csv"
    delta_rows_path = output_dir / "steered_delta_rows.csv"
    delta_matrix_path = output_dir / "delta_matrix_mean_logprob.csv"
    delta_summary_path = output_dir / "delta_by_steer_summary.csv"
    directional_summary_path = output_dir / "directional_match_by_steer.csv"
    emotion_gap_summary_path = output_dir / "emotion_gap_by_steer.csv"
    target_softmax_item_path = output_dir / "target_option_softmax_by_item.csv"
    target_softmax_summary_path = output_dir / "target_option_softmax_by_steer.csv"
    coverage_path = output_dir / "coverage_by_steer.csv"
    metadata_path = output_dir / "run_metadata.json"
    config_copy_path = output_dir / "config_used.yaml"

    raw_df = pd.DataFrame(list(baseline_rows) + list(steered_rows))
    if save_full_condition_artifacts:
        raw_df.to_csv(raw_csv_path, index=False)

    baseline = (
        pd.DataFrame(baseline_rows)
        .loc[:, ["item_id", "target_emotion", "target_sequence", "log_prob", "available"]]
        .rename(columns={"log_prob": "neutral_log_prob", "available": "neutral_available"})
    )
    steered = pd.DataFrame(steered_rows).merge(
        baseline,
        on=["item_id", "target_emotion", "target_sequence"],
        how="left",
    )
    steered["log_prob"] = pd.to_numeric(steered["log_prob"], errors="coerce")
    steered["neutral_log_prob"] = pd.to_numeric(steered["neutral_log_prob"], errors="coerce")
    steered["delta_log_prob"] = steered["log_prob"] - steered["neutral_log_prob"]
    if save_full_condition_artifacts:
        steered.to_csv(delta_rows_path, index=False)

    valid_delta = steered[steered["delta_log_prob"].notna()].copy()
    if save_full_condition_artifacts:
        runtime["build_delta_matrix"](
            df=valid_delta,
            emotions=[str(condition["emotion"])],
            target_emotions=TARGET_EMOTIONS,
        ).to_csv(delta_matrix_path, index=True)
        runtime["summarize_delta_by_steer"](valid_delta).to_csv(delta_summary_path, index=False)
        runtime["summarize_directional_match_by_steer"](valid_delta).to_csv(
            directional_summary_path, index=False
        )
        runtime["summarize_emotion_gap_by_steer"](steered).to_csv(
            emotion_gap_summary_path, index=False
        )

    target_softmax_item, target_softmax_summary = build_target_softmax_summary(
        baseline_rows=baseline_rows,
        steered_rows=steered_rows,
        emotion=str(condition["emotion"]),
        layer_1based=int(condition["layer_1based"]),
        control_layer=int(condition["control_layer"]),
        intensity=float(condition["intensity"]),
    )
    if save_full_condition_artifacts:
        target_softmax_item.to_csv(target_softmax_item_path, index=False)

    target_softmax_summary["emotion"] = str(condition["emotion"])
    target_softmax_summary["layer_1based"] = int(condition["layer_1based"])
    target_softmax_summary["control_layer"] = int(condition["control_layer"])
    target_softmax_summary["intensity"] = float(condition["intensity"])
    target_softmax_summary["previous_pd_layer_1based"] = int(
        PREVIOUS_PD_LAYERS[str(condition["emotion"])]
    )
    target_softmax_summary["is_previous_pd_layer"] = (
        target_softmax_summary["layer_1based"] == target_softmax_summary["previous_pd_layer_1based"]
    )
    target_softmax_summary["neutral_p_target_mean"] = float(
        pd.to_numeric(target_softmax_item["neutral_p_target"], errors="coerce").mean()
    )
    target_softmax_summary["neutral_margin_mean"] = float(
        pd.to_numeric(target_softmax_item["neutral_margin"], errors="coerce").mean()
    )
    target_softmax_summary["delta_p_target_mean"] = float(
        pd.to_numeric(target_softmax_item["delta_p_target"], errors="coerce").mean()
    )
    target_softmax_summary["delta_margin_mean"] = float(
        pd.to_numeric(target_softmax_item["delta_margin"], errors="coerce").mean()
    )
    target_softmax_summary.to_csv(target_softmax_summary_path, index=False)

    coverage = (
        steered.groupby("steer_emotion", as_index=False)
        .agg(
            total_pairs=("target_sequence", "size"),
            valid_delta_pairs=("delta_log_prob", lambda s: int(s.notna().sum())),
            valid_ratio=("delta_log_prob", lambda s: float(s.notna().mean())),
        )
        .sort_values("steer_emotion")
    )
    if save_full_condition_artifacts:
        coverage.to_csv(coverage_path, index=False)

    cfg = _base_config(model_path, stimulus_data_dir=str(stimulus_data_dir))
    cfg["control_layers"] = [int(condition["control_layer"])]
    cfg["intensity"] = float(condition["intensity"])
    cfg["run_tag"] = build_run_tag(condition)
    cfg["output_dir"] = str(output_dir.parent)
    cfg["output_config"]["save_full_condition_artifacts"] = bool(save_full_condition_artifacts)
    with config_copy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "model_path": model_path,
        "task_type": "self_report_emotion_options6",
        "steer_emotion": str(condition["emotion"]),
        "layer_1based": int(condition["layer_1based"]),
        "control_layer": int(condition["control_layer"]),
        "intensity": float(condition["intensity"]),
        "token_pos": "end",
        "previous_pd_layer_1based": int(PREVIOUS_PD_LAYERS[str(condition["emotion"])]),
        "is_previous_pd_layer": bool(
            int(condition["layer_1based"]) == int(PREVIOUS_PD_LAYERS[str(condition["emotion"])])
        ),
        "num_prompts": int(pd.DataFrame(steered_rows)["item_id"].nunique()),
        "num_target_options": len(TARGET_EMOTIONS),
        "raw_rows": int(len(raw_df)),
        "steered_rows": int(len(steered_rows)),
        "save_full_condition_artifacts": bool(save_full_condition_artifacts),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_target_softmax_summary(
    *,
    baseline_rows: Sequence[dict[str, Any]],
    steered_rows: Sequence[dict[str, Any]],
    emotion: str,
    layer_1based: int,
    control_layer: int,
    intensity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime = _runtime()
    steered_df = pd.DataFrame(steered_rows).copy()
    baseline_df = pd.DataFrame(baseline_rows).copy()

    target_softmax_item = runtime["build_target_option_softmax_metrics"](steered_df)
    target_softmax_item_ex_neutral = runtime["build_target_option_softmax_metrics"](
        steered_df,
        excluded_target_emotions={"neutral"},
    ).rename(
        columns={
            "p_target": "p_target_ex_neutral",
            "delta_p_target_vs_top_p_non_target": "margin_ex_neutral",
            "is_target_top_rank": "is_target_top_rank_ex_neutral",
            "n_options": "n_options_ex_neutral",
        }
    ).loc[
        :,
        [
            "item_id",
            "steer_emotion",
            "p_target_ex_neutral",
            "margin_ex_neutral",
            "is_target_top_rank_ex_neutral",
            "n_options_ex_neutral",
        ],
    ]
    baseline_df["steer_emotion"] = str(emotion)
    baseline_softmax = runtime["build_target_option_softmax_metrics"](baseline_df)
    baseline_softmax_ex_neutral = runtime["build_target_option_softmax_metrics"](
        baseline_df,
        excluded_target_emotions={"neutral"},
    )
    baseline_map = baseline_softmax.rename(
        columns={
            "p_target": "neutral_p_target",
            "delta_p_target_vs_top_p_non_target": "neutral_margin",
            "is_target_top_rank": "neutral_is_target_top_rank",
        }
    ).loc[:, ["item_id", "neutral_p_target", "neutral_margin", "neutral_is_target_top_rank"]]
    baseline_map_ex_neutral = baseline_softmax_ex_neutral.rename(
        columns={
            "p_target": "neutral_p_target_ex_neutral",
            "delta_p_target_vs_top_p_non_target": "neutral_margin_ex_neutral",
            "is_target_top_rank": "neutral_is_target_top_rank_ex_neutral",
            "n_options": "neutral_n_options_ex_neutral",
        }
    ).loc[
        :,
        [
            "item_id",
            "neutral_p_target_ex_neutral",
            "neutral_margin_ex_neutral",
            "neutral_is_target_top_rank_ex_neutral",
            "neutral_n_options_ex_neutral",
        ],
    ]

    target_softmax_item["emotion"] = str(emotion)
    target_softmax_item = target_softmax_item.merge(baseline_map, on="item_id", how="left")
    target_softmax_item = target_softmax_item.merge(
        target_softmax_item_ex_neutral.drop(columns=["steer_emotion"]),
        on="item_id",
        how="left",
    )
    target_softmax_item = target_softmax_item.merge(
        baseline_map_ex_neutral,
        on="item_id",
        how="left",
    )
    target_softmax_item["delta_p_target"] = (
        pd.to_numeric(target_softmax_item["p_target"], errors="coerce")
        - pd.to_numeric(target_softmax_item["neutral_p_target"], errors="coerce")
    )
    target_softmax_item["delta_margin"] = (
        pd.to_numeric(
            target_softmax_item["delta_p_target_vs_top_p_non_target"],
            errors="coerce",
        )
        - pd.to_numeric(target_softmax_item["neutral_margin"], errors="coerce")
    )
    target_softmax_item["delta_p_target_ex_neutral"] = (
        pd.to_numeric(target_softmax_item["p_target_ex_neutral"], errors="coerce")
        - pd.to_numeric(target_softmax_item["neutral_p_target_ex_neutral"], errors="coerce")
    )
    target_softmax_item["delta_margin_ex_neutral"] = (
        pd.to_numeric(target_softmax_item["margin_ex_neutral"], errors="coerce")
        - pd.to_numeric(target_softmax_item["neutral_margin_ex_neutral"], errors="coerce")
    )

    target_softmax_summary = runtime["summarize_target_option_softmax_by_steer"](
        target_softmax_item.loc[
            :,
            [
                "item_id",
                "steer_emotion",
                "p_target",
                "delta_p_target_vs_top_p_non_target",
                "is_target_top_rank",
            ],
        ]
    )
    target_softmax_summary["emotion"] = str(emotion)
    target_softmax_summary["layer_1based"] = int(layer_1based)
    target_softmax_summary["control_layer"] = int(control_layer)
    target_softmax_summary["intensity"] = float(intensity)
    target_softmax_summary["previous_pd_layer_1based"] = int(PREVIOUS_PD_LAYERS[str(emotion)])
    target_softmax_summary["is_previous_pd_layer"] = (
        target_softmax_summary["layer_1based"] == target_softmax_summary["previous_pd_layer_1based"]
    )
    target_softmax_summary["neutral_p_target_mean"] = float(
        pd.to_numeric(target_softmax_item["neutral_p_target"], errors="coerce").mean()
    )
    target_softmax_summary["neutral_margin_mean"] = float(
        pd.to_numeric(target_softmax_item["neutral_margin"], errors="coerce").mean()
    )
    target_softmax_summary["delta_p_target_mean"] = float(
        pd.to_numeric(target_softmax_item["delta_p_target"], errors="coerce").mean()
    )
    target_softmax_summary["delta_margin_mean"] = float(
        pd.to_numeric(target_softmax_item["delta_margin"], errors="coerce").mean()
    )
    target_softmax_summary["p_target_ex_neutral_mean"] = float(
        pd.to_numeric(target_softmax_item["p_target_ex_neutral"], errors="coerce").mean()
    )
    target_softmax_summary["margin_ex_neutral_mean"] = float(
        pd.to_numeric(target_softmax_item["margin_ex_neutral"], errors="coerce").mean()
    )
    target_softmax_summary["neutral_p_target_ex_neutral_mean"] = float(
        pd.to_numeric(target_softmax_item["neutral_p_target_ex_neutral"], errors="coerce").mean()
    )
    target_softmax_summary["neutral_margin_ex_neutral_mean"] = float(
        pd.to_numeric(target_softmax_item["neutral_margin_ex_neutral"], errors="coerce").mean()
    )
    target_softmax_summary["delta_p_target_ex_neutral_mean"] = float(
        pd.to_numeric(target_softmax_item["delta_p_target_ex_neutral"], errors="coerce").mean()
    )
    target_softmax_summary["delta_margin_ex_neutral_mean"] = float(
        pd.to_numeric(target_softmax_item["delta_margin_ex_neutral"], errors="coerce").mean()
    )
    return target_softmax_item, target_softmax_summary


def _collect_existing_summary(output_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(output_root.glob("*/target_option_softmax_by_steer.csv")):
        rows.append(pd.read_csv(path))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _set_benchmark_sample_limit(benchmark: Any, sample_limit: int | None) -> None:
    if sample_limit is None:
        return
    benchmark.sample_limit = int(sample_limit)


def run(
    *,
    model_path: str,
    stimulus_data_dir: str,
    output_root: Path,
    chunk_size: int,
    limit_conditions: int | None,
    skip_existing: bool,
    shard_index: int,
    num_shards: int,
    sample_limit: int | None,
    emotions: Sequence[str] | None,
    layers: Sequence[int] | None,
    intensities: Sequence[float] | None,
    tensor_parallel_size: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = _runtime()
    import torch

    runtime["repe_pipeline_registry"]()

    cfg = _base_config(
        model_path,
        tensor_parallel_size=int(tensor_parallel_size),
        stimulus_data_dir=str(stimulus_data_dir),
    )
    loading_config = runtime["_build_loading_config"](cfg, model_path=model_path)
    benchmark = runtime["_build_benchmark_config"](cfg)
    _set_benchmark_sample_limit(benchmark, sample_limit)
    repe_config = runtime["get_repe_eng_config"](model_path, yaml_config=cfg["repe_eng_config"])

    model_for_reader, tokenizer_for_reader, _, processor = runtime["setup_model_and_tokenizer"](
        loading_config,
        from_vllm=False,
    )
    all_hidden_layers = list(
        range(-1, -runtime["ModelLayerDetector"].num_layers(model_for_reader) - 1, -1)
    )
    emotion_rep_readers = runtime["load_emotion_readers"](
        repe_config,
        model_for_reader,
        tokenizer_for_reader,
        all_hidden_layers,
        processor,
        False,
    )
    del model_for_reader
    torch.cuda.empty_cache()

    model, tokenizer, prompt_format, _ = runtime["setup_model_and_tokenizer"](
        loading_config, from_vllm=True
    )
    rep_control_hook = runtime["RepControlVLLMHook"](
        model=model,
        tokenizer=tokenizer,
        layers=all_hidden_layers,
        block_name=repe_config.get("block_name", "decoder_block"),
        control_method=repe_config.get("control_method", "reading_vec"),
    )
    prompt_items = runtime["_load_prompt_items"](
        benchmark=benchmark,
        prompt_format=prompt_format,
        enable_thinking=False,
        user_messages=runtime["_resolve_benchmark_user_messages"](cfg),
    )
    target_pairs = runtime["_flatten_targets"](runtime["DEFAULT_LEXICAL_TARGETS"], TARGET_EMOTIONS)

    baseline_rows = _compute_rows_for_condition(
        rep_control_hook=rep_control_hook,
        tokenizer=tokenizer,
        prompt_items=prompt_items,
        target_pairs=target_pairs,
        steer_emotion="neutral",
        steer_intensity=0.0,
        activations=None,
        chunk_size=chunk_size,
    )
    save_full_condition_artifacts = bool(
        ((cfg.get("output_config") or {}).get("save_full_condition_artifacts", False))
    )
    if save_full_condition_artifacts:
        baseline_dir = output_root / "_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(baseline_rows).to_csv(baseline_dir / "raw_sequence_logprobs.csv", index=False)

    num_hidden_layers = _load_num_hidden_layers(model_path)
    grid = build_condition_grid(
        num_hidden_layers=num_hidden_layers,
        emotions=emotions,
        intensities=intensities,
        layers=layers,
    )
    if limit_conditions is not None:
        grid = grid[: int(limit_conditions)]
    grid = select_shard(grid=grid, shard_index=int(shard_index), num_shards=int(num_shards))

    for condition in grid:
        run_dir = condition_output_dir(output_root, condition)
        summary_path = run_dir / "target_option_softmax_by_steer.csv"
        metadata_path = run_dir / "run_metadata.json"
        if skip_existing and summary_path.exists() and metadata_path.exists():
            print(f"[skip] {run_dir}")
            continue

        activations = runtime["_build_control_activations"](
            emotion_rep_readers=emotion_rep_readers,
            control_layers=[int(condition["control_layer"])],
            emotion=str(condition["emotion"]),
            intensity=float(condition["intensity"]),
        )
        steered_rows = _compute_rows_for_condition(
            rep_control_hook=rep_control_hook,
            tokenizer=tokenizer,
            prompt_items=prompt_items,
            target_pairs=target_pairs,
            steer_emotion=str(condition["emotion"]),
            steer_intensity=float(condition["intensity"]),
            activations=activations,
            chunk_size=chunk_size,
        )
        _write_condition_outputs(
            model_path=model_path,
            stimulus_data_dir=str(stimulus_data_dir),
            output_dir=run_dir,
            baseline_rows=baseline_rows,
            steered_rows=steered_rows,
            condition=condition,
            save_full_condition_artifacts=save_full_condition_artifacts,
        )

    summary = _collect_existing_summary(output_root)
    if not summary.empty:
        summary.sort_values(["emotion", "layer_1based", "intensity"], inplace=True)
        summary.to_csv(output_root / "condition_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--limit-conditions", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--emotions", type=str, default=None)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--intensities", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--stimulus-data-dir", type=str, default=DEFAULT_STIMULUS_DATA_DIR)
    args = parser.parse_args()

    output_root = resolve_output_root(str(args.model_path), args.output_root)

    run(
        model_path=str(args.model_path),
        stimulus_data_dir=str(args.stimulus_data_dir),
        output_root=output_root,
        chunk_size=int(args.chunk_size),
        limit_conditions=args.limit_conditions,
        skip_existing=bool(args.skip_existing),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        sample_limit=args.sample_limit,
        emotions=parse_csv_arg(args.emotions, cast=str),
        layers=parse_csv_arg(args.layers, cast=int),
        intensities=parse_csv_arg(args.intensities, cast=float),
        tensor_parallel_size=int(args.tensor_parallel_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
