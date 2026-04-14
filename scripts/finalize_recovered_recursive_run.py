#!/usr/bin/env python3
# Purpose: finalize a recovered recursive resource-pipeline run by merging the completed latest round and writing final recursive artifacts.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emotion_experiment_engine.resource_recursive_workflow import (
    _has_schedulable_work,
    _load_report,
    _materialize_final_report,
    _write_json,
    advance_resource_round_state,
    merge_round_reports_for_state,
)


def _sorted_round_dirs(rounds_dir: Path) -> list[Path]:
    return sorted(
        [path for path in rounds_dir.iterdir() if path.is_dir() and path.name.startswith("round_")],
        key=lambda path: path.name,
    )


def _resource_gpus_from_round_dir(round_dir: Path) -> int | None:
    if "_g" not in round_dir.name:
        return None
    suffix = round_dir.name.rsplit("_g", 1)[1]
    digits = "".join(ch for ch in suffix if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def finalize_recovered_run(*, recovery_root: Path | str) -> Path:
    root = Path(recovery_root).expanduser().resolve()
    meta_dir = root / "meta"
    rounds_dir = root / "rounds"
    final_dir = root / "final"
    pipeline_config_path = meta_dir / "pipeline_config.json"
    if not pipeline_config_path.exists():
        raise FileNotFoundError(f"Missing pipeline config: {pipeline_config_path}")

    pipeline_config = json.loads(pipeline_config_path.read_text(encoding="utf-8"))
    round_dirs = _sorted_round_dirs(rounds_dir)
    if not round_dirs:
        raise ValueError("Recovered run must contain at least one round directory")

    latest_round_dir = round_dirs[-1]
    latest_round_manifest_path = latest_round_dir / "resource_round_manifest.json"
    if not latest_round_manifest_path.exists():
        raise FileNotFoundError(f"Missing latest round manifest: {latest_round_manifest_path}")

    latest_round_manifest = json.loads(latest_round_manifest_path.read_text(encoding="utf-8"))
    if len(round_dirs) == 1:
        planning_report = (root / "source" / "source_report.json").expanduser().resolve()
    else:
        previous_round_dir = round_dirs[-2]
        previous_round_manifest_path = previous_round_dir / "round_manifest.json"
        if not previous_round_manifest_path.exists():
            raise FileNotFoundError(f"Missing previous round manifest: {previous_round_manifest_path}")
        previous_round_manifest = json.loads(previous_round_manifest_path.read_text(encoding="utf-8"))
        planning_report = Path(str(previous_round_manifest["next_planning_report"])).expanduser().resolve()
    carry_forward_report = Path(str(latest_round_manifest["carry_forward_report"])).expanduser().resolve()
    shard_reports = [
        Path(str(report_path)).expanduser().resolve()
        for report_path in latest_round_manifest.get("shard_reports", [])
    ]
    if not shard_reports:
        raise ValueError("Latest round manifest must contain at least one shard report")

    current_round_gpu_count = _resource_gpus_from_round_dir(latest_round_dir)
    if current_round_gpu_count is None:
        raise ValueError(f"Could not infer resource GPU count from round dir: {latest_round_dir}")
    max_resource_gpus = int(pipeline_config["max_resource_gpus"])
    final_series_name = str(pipeline_config.get("final_merged_series_name", "resource_pipeline_final"))

    merged_state_report = merge_round_reports_for_state(
        planning_report,
        carry_forward_report=carry_forward_report,
        shard_reports=shard_reports,
        merged_output_dir=latest_round_dir,
        merged_series_name=f"{final_series_name}_state_r{len(round_dirs):02d}_g{current_round_gpu_count}",
    )
    next_planning_report = advance_resource_round_state(
        merged_state_report,
        output_dir=latest_round_dir,
        merged_series_name=f"{final_series_name}_planning_r{len(round_dirs) + 1:02d}",
        current_round_gpu_count=current_round_gpu_count,
        max_resource_gpus=max_resource_gpus,
    )

    round_manifest = dict(latest_round_manifest)
    round_manifest["round_index"] = len(round_dirs)
    round_manifest["merged_state_report"] = str(merged_state_report)
    round_manifest["next_planning_report"] = str(next_planning_report)
    _write_json(latest_round_dir / "round_manifest.json", round_manifest)

    next_payload = _load_report(next_planning_report)
    next_round_gpu_count = int(
        next_payload.get("series_config", {}).get("current_round_gpu_count", current_round_gpu_count) or current_round_gpu_count
    )
    if _has_schedulable_work(
        next_planning_report,
        current_round_gpu_count=next_round_gpu_count,
        max_resource_gpus=max_resource_gpus,
    ):
        raise ValueError(
            "Recovered run still has schedulable work for a later resource tier; "
            "resume the recursive pipeline instead of finalizing it."
        )

    final_report = _materialize_final_report(next_planning_report, final_dir)
    summary = {
        "source_report": str(root / "source" / "source_report.json"),
        "planning_report": str(next_planning_report),
        "gpu_pool": list(pipeline_config.get("gpu_pool", [])),
        "resumed_from_pipeline_root": pipeline_config.get("resumed_from_pipeline_root"),
        "rounds": [],
        "final_report": str(final_report),
    }

    for idx, round_dir in enumerate(round_dirs, start=1):
        manifest_path = round_dir / "round_manifest.json"
        if manifest_path.exists():
            round_entry = {
                "round_index": idx,
                "resource_gpus": _resource_gpus_from_round_dir(round_dir),
                "round_dir": str(round_dir),
                "manifest_path": str(manifest_path),
            }
            manifest_data = _load_report(manifest_path)
            if "merged_state_report" in manifest_data:
                round_entry["merged_state_report"] = str(manifest_data["merged_state_report"])
            if "next_planning_report" in manifest_data:
                round_entry["next_planning_report"] = str(manifest_data["next_planning_report"])
            if manifest_data.get("recovered"):
                round_entry["recovered"] = True
            summary["rounds"].append(round_entry)

    _write_json(meta_dir / "summary.json", summary)
    _write_json(root / "manifest.json", summary)
    return final_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a recovered recursive resource pipeline run."
    )
    parser.add_argument("--recovery-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    final_report = finalize_recovered_run(recovery_root=args.recovery_root)
    print(final_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
