"""
Responsible: auto_experiments/task_similarity/migrate_emotion_pd_delta_similarity_results.py
Purpose: Migrate legacy similarity result layout to the new run_id-first layout.

Legacy layout (per emotion run):
  results/anger_pd_delta_similarity/<model>/<emotion>/<timestamp>/seed_<seed>/

New layout (date/run_id is the identifier):
  results/anger_pd_delta_similarity/<run_id>/<model>/<emotion>/seed_<seed>/

This script moves directories in-place (no recomputation).
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LegacyRun:
    metadata_path: Path
    model: str
    emotion: str
    run_id: str
    seed_dir: Path


def is_legacy_metadata_path(metadata_path: Path, *, root: Path) -> bool:
    """
    Legacy metadata path shape:
      <root>/<model>/<emotion>/<run_id>/seed_<seed>/metadata.json

    New metadata path shape:
      <root>/<run_id>/<model>/<emotion>/seed_<seed>/metadata.json
    """
    p = Path(metadata_path)
    if p.name != "metadata.json":
        return False
    try:
        rel = p.relative_to(Path(root))
    except Exception:
        return False
    # legacy: model/emotion/run_id/seed_x/metadata => 5 parts
    # new:    run_id/model/emotion/seed_x/metadata => 5 parts
    # Need to disambiguate by checking where seed dir sits:
    # legacy: rel.parts[-2] == seed_x and rel.parts[-3] == run_id and rel.parts[-4] == emotion
    parts = rel.parts
    if len(parts) != 5:
        return False
    if not parts[-2].startswith("seed_"):
        return False
    # If it's legacy, the seed dir's parent is run_id AND that parent's parent is emotion.
    # If it's new, the seed dir's parent is emotion AND that parent's parent is model.
    # So: legacy => parts[-3] is run_id and parts[-4] is emotion.
    #      new    => parts[-3] is emotion and parts[-4] is model.
    # We classify as legacy if the third-from-end looks like a datetime run_id.
    run_id = parts[-3]
    # Heuristic: run_id like YYYYMMDD_HHMMSS.
    if len(run_id) == 15 and run_id[8] == "_" and run_id.replace("_", "").isdigit():
        return True
    return False


def map_legacy_metadata_to_dest_seed_dir(metadata_path: Path, *, root: Path) -> Path:
    """
    Convert:
      <root>/<model>/<emotion>/<run_id>/seed_<seed>/metadata.json
    To:
      <root>/<run_id>/<model>/<emotion>/seed_<seed>
    """
    p = Path(metadata_path)
    rel = p.relative_to(Path(root))
    model, emotion, run_id, seed_dir, _ = rel.parts
    return Path(root) / run_id / model / emotion / seed_dir


def iter_legacy_runs(root: Path) -> Iterable[LegacyRun]:
    root = Path(root)
    for meta in root.rglob("metadata.json"):
        if not is_legacy_metadata_path(meta, root=root):
            continue
        rel = meta.relative_to(root)
        model, emotion, run_id, seed_dir, _ = rel.parts
        yield LegacyRun(
            metadata_path=meta,
            model=str(model),
            emotion=str(emotion),
            run_id=str(run_id),
            seed_dir=meta.parent,
        )


def migrate(root: Path, *, dry_run: bool = False) -> Dict[str, object]:
    root = Path(root)
    moved: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    by_run_id: Dict[str, List[LegacyRun]] = {}
    for run in iter_legacy_runs(root):
        by_run_id.setdefault(run.run_id, []).append(run)

    for run_id, runs in sorted(by_run_id.items()):
        for run in runs:
            src = run.seed_dir
            dest = map_legacy_metadata_to_dest_seed_dir(run.metadata_path, root=root)
            if dest.exists():
                skipped.append(
                    {
                        "reason": "dest_exists",
                        "src": str(src),
                        "dest": str(dest),
                    }
                )
                continue
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            moved.append({"src": str(src), "dest": str(dest)})

        # Write a minimal config.json if missing.
        cfg_path = root / run_id / "config.json"
        if not cfg_path.exists():
            payload = {
                "run_id": run_id,
                "created_by": "migrate_emotion_pd_delta_similarity_results.py",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Migrated from legacy layout; fields may be partial.",
                "moved_entries": [m for m in moved if f"/{run_id}/" in m["dest"]],
            }
            if not dry_run:
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"root": str(root), "dry_run": bool(dry_run), "moved": moved, "skipped": skipped}


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate legacy emotion_pd_delta_similarity results to run_id-first layout.")
    p.add_argument(
        "--root",
        default="auto_experiments/task_similarity/results/anger_pd_delta_similarity",
        help="Root results directory to migrate.",
    )
    p.add_argument("--dry_run", action="store_true", help="Report moves without changing filesystem.")
    args = p.parse_args()

    report = migrate(Path(args.root), dry_run=bool(args.dry_run))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
