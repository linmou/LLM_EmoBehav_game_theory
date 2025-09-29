"""Utilities for running SWE-bench harness on deferred experiment outputs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_KEY_INSTANCE_ID = "instance_id"
_KEY_MODEL = "model_name_or_path"
_KEY_PREDICTION = "model_patch"


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_model_name(model_path: str) -> str:
    return model_path.replace("/", "__")


def _ensure_predictions_ready(
    original_path: Path,
    staging_dir: Path,
    model_name: str,
) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = staging_dir / original_path.name
    prepared_lines: List[str] = []

    for line in original_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record[_KEY_MODEL] = model_name
        if _KEY_INSTANCE_ID not in record:
            raise ValueError(
                f"Prediction missing '{_KEY_INSTANCE_ID}' in {original_path}"
            )
        if _KEY_PREDICTION not in record:
            raise ValueError(
                f"Prediction missing '{_KEY_PREDICTION}' in {original_path}"
            )
        prepared_lines.append(json.dumps(record, ensure_ascii=False))

    prepared_path.write_text("\n".join(prepared_lines) + "\n", encoding="utf-8")
    return prepared_path.resolve()


def _load_run_metadata(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    raw_path = run_dir / "raw_results.json"
    raw_records = _read_json(raw_path)
    run_map: Dict[str, Dict[str, Any]] = {}

    for record in raw_records:
        meta = record.get("metadata") or {}
        run_id = meta.get("run_id")
        predictions_path = meta.get("predictions_path")
        if not run_id or not predictions_path:
            continue
        entry = run_map.setdefault(
            run_id,
            {
                "emotion": record.get("emotion"),
                "intensity": record.get("intensity"),
                "repeat_id": record.get("repeat_id", 0),
                "predictions_path": predictions_path,
            },
        )
        # Preserve first occurrence (metadata identical across samples)
        entry.setdefault("emotion", record.get("emotion"))
    return run_map


def _resolve_predictions_path(predictions_path: str, run_dir: Path, run_id: str) -> Path:
    path = Path(predictions_path)
    if path.exists():
        return path
    fallback = run_dir / "predictions" / f"{run_id}.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Cannot locate predictions file for run '{run_id}': {predictions_path}"
    )


def _build_harness_command(
    *,
    python_executable: str,
    dataset_name: str,
    split: str,
    prepared_path: Path,
    run_id: str,
    report_dir: Path,
    max_workers: Optional[int],
    extra_args: Optional[Iterable[str]],
) -> List[str]:
    command: List[str] = [
        python_executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(prepared_path),
        "--run_id",
        run_id,
        "--report_dir",
        str(report_dir),
    ]
    if max_workers is not None:
        command.extend(["--max_workers", str(max_workers)])
    if extra_args:
        command.extend(list(extra_args))
    return command


def evaluate_swebench_run(
    *,
    run_dir: Path,
    swebench_repo: Path,
    dataset_name: str,
    split: str,
    results_root: Path,
    python_executable: str = "python",
    max_workers: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    swebench_repo = Path(swebench_repo)
    results_root = Path(results_root)

    exp_cfg = _read_json(run_dir / "experiment_config.json")
    model_path = exp_cfg.get("model_path", "unknown")
    model_slug = Path(model_path).name or "model"

    run_entries = _load_run_metadata(run_dir)
    if not run_entries:
        raise ValueError(f"No SWE-bench predictions found under {run_dir}")

    staging_dir = run_dir / "harness_inputs"
    report_dir = run_dir / "harness_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for run_id in sorted(run_entries.keys()):
        entry = run_entries[run_id]
        original_path = _resolve_predictions_path(entry["predictions_path"], run_dir, run_id)
        prepared_path = _ensure_predictions_ready(original_path, staging_dir, model_path)

        command = _build_harness_command(
            python_executable=python_executable,
            dataset_name=dataset_name,
            split=split,
            prepared_path=prepared_path,
            run_id=run_id,
            report_dir=report_dir,
            max_workers=max_workers,
            extra_args=extra_args,
        )
        subprocess.run(
            command,
            check=True,
            cwd=str(swebench_repo),
        )

        report_name = f"{_sanitize_model_name(model_path)}.{run_id}.json"
        report_path = report_dir / report_name
        if not report_path.exists():
            fallback_report = swebench_repo / report_name
            if fallback_report.exists():
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path.write_text(fallback_report.read_text(encoding="utf-8"), encoding="utf-8")
                fallback_report.unlink(missing_ok=True)
            else:
                logs_path = swebench_repo / "logs" / "run_evaluation" / run_id
                if logs_path.exists():
                    json_candidates = list(logs_path.rglob("*.json"))
                    json_candidates = [p for p in json_candidates if p.name == "report.json" or p.name.endswith(".json")]
                    if json_candidates:
                        source_report = sorted(json_candidates)[-1]
                        report_dir.mkdir(parents=True, exist_ok=True)
                        report_path.write_text(source_report.read_text(encoding="utf-8"), encoding="utf-8")
                        source_report.unlink(missing_ok=True)
        report = _read_json(report_path)
        total = int(report.get("total_instances", 0))
        resolved = int(report.get("resolved_instances", 0))
        pass_rate = resolved / total if total else 0.0

        summaries.append(
            {
                "emotion": entry.get("emotion"),
                "intensity": entry.get("intensity"),
                "repeat_id": int(entry.get("repeat_id", 0)),
                "run_id": run_id,
                "predictions_path": str(original_path),
                "prepared_predictions_path": str(prepared_path),
                "harness_report_path": str(report_path),
                "resolved_instances": resolved,
                "total_instances": total,
                "pass_rate": pass_rate,
                "empty_patch_instances": int(report.get("empty_patch_instances", 0)),
                "error_instances": int(report.get("error_instances", 0)),
            }
        )

    manifest = {
        "model_path": model_path,
        "benchmark": exp_cfg.get("benchmark", {}),
        "dataset_name": dataset_name,
        "split": split,
        "evaluated_at": datetime.utcnow().isoformat(),
        "runs": summaries,
    }

    manifest_dir = results_root / model_slug
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_dir.name}_evaluation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest

__all__ = ["evaluate_swebench_run"]
