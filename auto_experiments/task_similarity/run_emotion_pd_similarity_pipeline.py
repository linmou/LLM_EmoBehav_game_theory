"""
Responsible: auto_experiments/task_similarity/run_emotion_pd_similarity_pipeline.py
Purpose: Orchestrate multi-emotion PD similarity + decision-impact pipeline (Python-first; bash wrapper stays thin).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from auto_experiments.task_similarity.pipeline_config_reader import read_pipeline_config


DEFAULT_OUTPUT_ROOT = Path("auto_experiments/task_similarity/results/emotion_pd_delta_similarity")
DEFAULT_HF_ROOT = Path("/data/home/jjl7137/huggingface_models")
DEFAULT_REP_READER_BASE = Path(
    "/data/home/jjl7137/LLM_EmoBehav_game_theory/neuro_manipulation/representation_storage"
)


def resolve_model_path(
    *,
    model_path: str,
    override: Optional[str],
    huggingface_root: Path = DEFAULT_HF_ROOT,
    is_dir: Callable[[Path], bool] = Path.is_dir,
) -> Path:
    if override:
        selected = Path(override)
        if not is_dir(selected):
            raise FileNotFoundError(f"Model path not found: {selected}")
        return selected

    selected = Path(model_path)
    if is_dir(selected):
        return selected

    marker = "/huggingface_models/"
    if marker in model_path:
        suffix = model_path.split(marker, 1)[1]
        cand = huggingface_root / suffix
        if is_dir(cand):
            return cand

    raise FileNotFoundError(
        f"Model path not found: {selected}\n"
        f"Pass a local path via: --model_path {huggingface_root}/..."
    )


def repreader_cache_path_from_args(
    *,
    args: Mapping[str, Any],
    base_dir: Path,
    dict_to_unique_code: Callable[[Mapping[str, Any]], str],
) -> Path:
    code = dict_to_unique_code(dict(args))
    expected = base_dir / f"emotion_rep_reader_{code[:10]}.pkl"
    if not expected.is_file():
        raise FileNotFoundError(f"Emotion RepReader pickle not found: {expected}")
    return expected


def repreader_cache_path_from_repe_config(
    repe_eng_config: Mapping[str, Any],
    *,
    num_layers: int,
    base_dir: Path,
    dict_to_unique_code: Callable[[Mapping[str, Any]], str],
    validate_multimodal: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Path:
    feasibility = validate_multimodal(repe_eng_config)
    if not feasibility.get("feasible", False):
        raise ValueError(f"Config not feasible for emotion readers: {feasibility.get('reasons')}")

    args = {
        "emotions": repe_eng_config["emotions"],
        "data_dir": repe_eng_config["data_dir"],
        "model_name_or_path": repe_eng_config["model_name_or_path"],
        "rep_token": repe_eng_config["rep_token"],
        "hidden_layers": list(range(-1, -int(num_layers) - 1, -1)),
        "n_difference": repe_eng_config["n_difference"],
        "direction_method": repe_eng_config["direction_method"],
        "experiment_mode": feasibility["mode"],
        "multimodal_intent": bool(repe_eng_config.get("multimodal_intent", False)),
        "emotion_data_seed": int(repe_eng_config.get("emotion_data_seed", 0)),
    }
    return repreader_cache_path_from_args(args=args, base_dir=base_dir, dict_to_unique_code=dict_to_unique_code)


def build_extra_pd_args(
    *,
    emotion_rep_reader: Path,
    pd_vectors_dir: Optional[Path],
    split_manifest: Optional[Path],
    save_tensors: bool,
    tensor_dtype: str,
) -> List[str]:
    args: List[str] = ["--emotion_rep_reader", str(emotion_rep_reader)]
    if pd_vectors_dir is not None:
        args += ["--pd_vectors_dir", str(pd_vectors_dir)]
    if split_manifest is not None:
        args += ["--split_manifest", str(split_manifest)]
    if save_tensors:
        args += ["--save_tensors", "--tensor_dtype", tensor_dtype]
    return args


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _run_module(module: str, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)


def _load_repe_eng_config_from_experiment_config(experiment_config_path: Path) -> Mapping[str, Any]:
    raw = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment_config.json must be a JSON object")
    repe = raw.get("repe_eng_config")
    if isinstance(repe, dict):
        return repe
    return raw


def _auto_select_pd_vectors(model_name: str) -> tuple[Optional[Path], Optional[Path]]:
    base = Path("auto_experiments/task_similarity/results/steering_vectors") / model_name
    if not base.is_dir():
        return None, None

    seed_dirs = sorted([p for p in base.rglob("seed_20") if p.is_dir()])
    chosen_seed = seed_dirs[-1] if seed_dirs else None
    if chosen_seed is None:
        manifests = sorted(base.rglob("split_manifest.json"))
        chosen_seed = manifests[-1].parent if manifests else None

    if chosen_seed is None:
        return None, None

    pd_vectors_dir = chosen_seed / "layer_vectors"
    split_manifest = chosen_seed / "split_manifest.json"
    return pd_vectors_dir if pd_vectors_dir.exists() else None, split_manifest if split_manifest.exists() else None


@dataclass(frozen=True)
class PipelineArgs:
    result_dir: Path
    model_path_override: Optional[str]
    max_length: int
    batch_size: int
    device_map: str
    split: str
    run_id: Optional[str]
    pd_vectors_dir: Optional[Path]
    split_manifest: Optional[Path]
    emotion_rep_reader: Optional[Path]
    save_tensors: bool
    tensor_dtype: str


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineArgs:
    p = argparse.ArgumentParser(
        description="Run multi-emotion PD similarity → decision-impact join → summary."
    )
    p.add_argument("--result_dir", required=True)
    p.add_argument("--model_path", dest="model_path_override", default=None)
    p.add_argument("--max_length", default=1024, type=int)
    p.add_argument("--batch_size", default=60, type=int)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--run_id", default=None)
    p.add_argument("--pd_vectors_dir", default=None)
    p.add_argument("--split_manifest", default=None)
    p.add_argument("--emotion_rep_reader", default=None)
    p.add_argument("--save_tensors", action="store_true", default=True)
    p.add_argument("--no_save_tensors", action="store_false", dest="save_tensors")
    p.add_argument("--tensor_dtype", default="float16", choices=["float16", "float32"])
    ns = p.parse_args(list(argv) if argv is not None else None)

    return PipelineArgs(
        result_dir=Path(ns.result_dir),
        model_path_override=ns.model_path_override,
        max_length=int(ns.max_length),
        batch_size=int(ns.batch_size),
        device_map=str(ns.device_map),
        split=str(ns.split),
        run_id=ns.run_id,
        pd_vectors_dir=Path(ns.pd_vectors_dir) if ns.pd_vectors_dir else None,
        split_manifest=Path(ns.split_manifest) if ns.split_manifest else None,
        emotion_rep_reader=Path(ns.emotion_rep_reader) if ns.emotion_rep_reader else None,
        save_tensors=bool(ns.save_tensors),
        tensor_dtype=str(ns.tensor_dtype),
    )


def run_pipeline(args: PipelineArgs) -> Path:
    exp_cfg_path = args.result_dir / "experiment_config.json"
    if not exp_cfg_path.is_file():
        raise FileNotFoundError(f"Missing {exp_cfg_path}")

    cfg = read_pipeline_config(exp_cfg_path)
    repe_eng_config = _load_repe_eng_config_from_experiment_config(exp_cfg_path)
    model_path_selected = resolve_model_path(
        model_path=cfg.model_path,
        override=args.model_path_override,
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = DEFAULT_OUTPUT_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    emotion_rep_reader = args.emotion_rep_reader
    if emotion_rep_reader is None:
        if not DEFAULT_REP_READER_BASE.is_dir():
            raise FileNotFoundError(f"RepReader base folder not found: {DEFAULT_REP_READER_BASE}")

        from neuro_manipulation.utils import dict_to_unique_code, validate_multimodal_experiment_feasibility
        from transformers import AutoConfig

        model_cfg = AutoConfig.from_pretrained(str(model_path_selected), trust_remote_code=True)
        num_layers = int(getattr(model_cfg, "num_hidden_layers"))

        emotion_rep_reader = repreader_cache_path_from_repe_config(
            repe_eng_config,
            num_layers=num_layers,
            base_dir=DEFAULT_REP_READER_BASE,
            dict_to_unique_code=dict_to_unique_code,
            validate_multimodal=validate_multimodal_experiment_feasibility,
        )
    if not emotion_rep_reader.is_file():
        raise FileNotFoundError(f"Emotion RepReader not found: {emotion_rep_reader}")

    pd_vectors_dir = args.pd_vectors_dir
    split_manifest = args.split_manifest
    if pd_vectors_dir is None or split_manifest is None:
        auto_pd_dir, auto_manifest = _auto_select_pd_vectors(model_path_selected.name)
        pd_vectors_dir = pd_vectors_dir or auto_pd_dir
        split_manifest = split_manifest or auto_manifest

    snapshot = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "result_dir": str(args.result_dir),
        "model_path_selected": str(model_path_selected),
        "emotions": ",".join(cfg.emotions),
        "intensities": ",".join(str(x) for x in cfg.intensities),
        "split": args.split,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device_map": args.device_map,
        "emotion_rep_reader": str(emotion_rep_reader),
        "pd_vectors_dir": str(pd_vectors_dir) if pd_vectors_dir else "",
        "split_manifest": str(split_manifest) if split_manifest else "",
        "save_tensors": args.save_tensors,
        "tensor_dtype": args.tensor_dtype,
    }
    (run_root / "config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    extra_pd_args = build_extra_pd_args(
        emotion_rep_reader=emotion_rep_reader,
        pd_vectors_dir=pd_vectors_dir,
        split_manifest=split_manifest,
        save_tensors=args.save_tensors,
        tensor_dtype=args.tensor_dtype,
    )

    intensities_csv = ",".join(str(x) for x in cfg.intensities)
    for emo in cfg.emotions:
        raw_results_path = args.result_dir / "raw_results.json"
        extra_similarity_args: List[str] = []
        if raw_results_path.is_file():
            extra_similarity_args += ["--raw_results_path", str(raw_results_path)]

        sim_args = [
            "--emotion",
            emo,
            "--split",
            args.split,
            "--run_id",
            run_id,
            "--intensities",
            intensities_csv,
            "--model",
            str(model_path_selected),
            "--max_length",
            str(args.max_length),
            "--batch_size",
            str(args.batch_size),
            "--device_map",
            args.device_map,
            "--output_root",
            str(DEFAULT_OUTPUT_ROOT),
            *extra_pd_args,
            *extra_similarity_args,
        ]

        sim_proc = _run_module("auto_experiments.task_similarity.emotion_pd_delta_similarity", sim_args)
        sim_stdout = sim_proc.stdout.strip()
        if not sim_stdout:
            raise RuntimeError("Similarity runner did not print a run directory to stdout")
        sim_dir = Path(sim_stdout)

        _run_module(
            "auto_experiments.task_similarity.analyze_similarity_decision_impact",
            ["--similarity_run_dir", str(sim_dir), "--result_dir", str(args.result_dir), "--emotion", emo],
        )
        _run_module(
            "auto_experiments.task_similarity.summarize_similarity_decision_impact",
            ["--impact_dir", str(sim_dir / "decision_impact" / emo), "--top_k", "10", "--last_k", "5"],
        )

    return run_root


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_root = run_pipeline(args)
    print(str(run_root))


if __name__ == "__main__":
    main()
