"""
Responsible: auto_experiments/task_similarity/group_emotion_pd_delta_similarity_runs.py
Purpose: Group per-emotion similarity runs into bash-level sessions based on time proximity.

After earlier migrations, you can have per-emotion folders like:
  results/anger_pd_delta_similarity/<run_id>/<model>/<emotion>/seed_<seed>/

But for usability, we want bash-level run_id (same for all emotions in a run):
  results/anger_pd_delta_similarity/<bash_run_id>/<model>/<emotion>/seed_<seed>/

This script groups run_ids by time window and merges them by moving emotion folders
into the earliest run_id directory of each group. It also rewrites metadata.json
to update `run_id` and preserve `original_run_id`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


def parse_run_id(run_id: str) -> datetime:
    if not RUN_ID_RE.match(str(run_id)):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return datetime.strptime(str(run_id), "%Y%m%d_%H%M%S")


def group_run_ids_by_time_window(run_ids: Sequence[str], *, window_seconds: int) -> List[List[str]]:
    if int(window_seconds) <= 0:
        raise ValueError("window_seconds must be positive")
    cleaned = [str(r) for r in run_ids if RUN_ID_RE.match(str(r))]
    cleaned.sort(key=parse_run_id)
    groups: List[List[str]] = []
    for rid in cleaned:
        if not groups:
            groups.append([rid])
            continue
        prev = groups[-1][-1]
        dt_prev = parse_run_id(prev)
        dt_cur = parse_run_id(rid)
        if (dt_cur - dt_prev).total_seconds() <= int(window_seconds):
            groups[-1].append(rid)
        else:
            groups.append([rid])
    return groups


def rewrite_metadata_run_id(meta_path: Path, *, new_run_id: str) -> None:
    payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    old = payload.get("run_id")
    if old and str(old) != str(new_run_id):
        payload.setdefault("original_run_id", str(old))
    payload["run_id"] = str(new_run_id)
    Path(meta_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class SeedEvent:
    run_id: str  # time identifier used for grouping
    model: str
    emotion: str
    seed: str
    seed_dir: Path  # current location: .../<some_run_dir>/<model>/<emotion>/seed_x


def _infer_original_run_id_from_decision_impact(seed_dir: Path, emotion: str) -> Optional[str]:
    meta = seed_dir / "decision_impact" / str(emotion) / "metadata.json"
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    sim_dir = payload.get("similarity_run_dir")
    if not isinstance(sim_dir, str):
        return None
    # legacy similarity_run_dir included: .../<emotion>/<timestamp>/seed_<seed>
    parts = Path(sim_dir).parts
    # Find ".../<emotion>/<run_id>/seed_x"
    for i in range(len(parts) - 2):
        if parts[i] == str(emotion) and RUN_ID_RE.match(parts[i + 1]) and parts[i + 2].startswith("seed_"):
            return str(parts[i + 1])
    return None


def iter_seed_events(root: Path) -> Iterable[SeedEvent]:
    root = Path(root)
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        if not RUN_ID_RE.match(run_dir.name):
            continue
        for meta in run_dir.rglob("metadata.json"):
            # expected: <run_id>/<model>/<emotion>/seed_x/metadata.json
            try:
                rel = meta.relative_to(run_dir)
            except Exception:
                continue
            parts = rel.parts
            if len(parts) != 4:
                continue
            model, emotion, seed_dir, _ = parts
            if not str(seed_dir).startswith("seed_"):
                continue
            seed_path = meta.parent
            # Prefer decision_impact pointer (most reliable for reconstructing original per-emotion timestamp),
            # else fall back to metadata["original_run_id"], else folder name.
            run_id = _infer_original_run_id_from_decision_impact(seed_path, emotion=str(emotion))
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                if not (isinstance(run_id, str) and RUN_ID_RE.match(run_id)):
                    run_id = payload.get("original_run_id")
            except Exception:
                pass
            if not (isinstance(run_id, str) and RUN_ID_RE.match(run_id)):
                run_id = run_dir.name

            yield SeedEvent(
                run_id=str(run_id),
                model=str(model),
                emotion=str(emotion),
                seed=str(seed_dir),
                seed_dir=seed_path,
            )


def group_events_into_sessions(
    events: Sequence[SeedEvent], *, window_seconds: int, start_emotion: str = "emotion"
) -> List[List[SeedEvent]]:
    if int(window_seconds) <= 0:
        raise ValueError("window_seconds must be positive")
    start_emotion = str(start_emotion)
    evs = sorted(list(events), key=lambda e: parse_run_id(e.run_id))
    out: List[List[SeedEvent]] = []
    for e in evs:
        if not out:
            out.append([e])
            continue
        cur = out[-1]
        last = cur[-1]
        dt_gap = (parse_run_id(e.run_id) - parse_run_id(last.run_id)).total_seconds()
        emotions_in_cur = {x.emotion for x in cur}
        if (
            dt_gap > int(window_seconds)
            or e.emotion in emotions_in_cur
            or (e.emotion == start_emotion and len(cur) > 0)
        ):
            out.append([e])
        else:
            cur.append(e)
    return out


def regroup(root: Path, *, window_seconds: int = 10 * 60, start_emotion: str = "emotion", dry_run: bool = False) -> Dict[str, object]:
    root = Path(root)

    events: List[SeedEvent] = list(iter_seed_events(root))
    # Group separately per (model, seed) so we don't mix models or seeds.
    by_key: Dict[Tuple[str, str], List[SeedEvent]] = {}
    for e in events:
        by_key.setdefault((e.model, e.seed), []).append(e)

    moved: List[Dict[str, str]] = []
    groups_out: List[Dict[str, object]] = []

    for (model, seed), evs in sorted(by_key.items()):
        sessions = group_events_into_sessions(evs, window_seconds=int(window_seconds), start_emotion=str(start_emotion))
        for sess in sessions:
            if len(sess) <= 1:
                continue
            bash_run_id = sess[0].run_id
            group_rec = {
                "bash_run_id": bash_run_id,
                "model": model,
                "seed": seed,
                "members": [e.run_id for e in sess],
                "emotions": [e.emotion for e in sess],
            }
            groups_out.append(group_rec)

            bash_model_dir = root / bash_run_id / model
            for e in sess[1:]:
                dest_seed_dir = bash_model_dir / e.emotion / seed
                if dest_seed_dir.exists():
                    continue
                if not dry_run:
                    dest_seed_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(e.seed_dir), str(dest_seed_dir))
                    meta_path = dest_seed_dir / "metadata.json"
                    if meta_path.exists():
                        rewrite_metadata_run_id(meta_path, new_run_id=bash_run_id)
                moved.append(
                    {
                        "bash_run_id": bash_run_id,
                        "src_seed_dir": str(e.seed_dir),
                        "dest_seed_dir": str(dest_seed_dir),
                    }
                )

            cfg_path = root / bash_run_id / "config.json"
            if not cfg_path.exists() and not dry_run:
                payload = {
                    "run_id": bash_run_id,
                    "created_by": "group_emotion_pd_delta_similarity_runs.py",
                    "window_seconds": int(window_seconds),
                    "start_emotion": str(start_emotion),
                    "model": model,
                    "seed": seed,
                    "member_run_ids": [e.run_id for e in sess],
                    "emotions": [e.emotion for e in sess],
                    "note": "Grouped legacy per-emotion runs by time window; fields may be partial.",
                }
                cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Cleanup: remove run_id dirs that became empty or config-only.
    cleaned: List[str] = []
    for d in root.iterdir():
        if not d.is_dir() or not RUN_ID_RE.match(d.name):
            continue
        # Remove empty directories under run folders (e.g., empty emotion dirs after moving seed_ dirs).
        if not dry_run:
            changed = True
            while changed:
                changed = False
                for sub in sorted([p for p in d.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
                    try:
                        if not any(sub.iterdir()):
                            sub.rmdir()
                            changed = True
                    except Exception:
                        pass
        items = list(d.iterdir())
        if not items:
            if not dry_run:
                d.rmdir()
            cleaned.append(str(d))
            continue
        if len(items) == 1 and items[0].name == "config.json":
            if not dry_run:
                items[0].unlink(missing_ok=True)
                d.rmdir()
            cleaned.append(str(d))

    return {
        "root": str(root),
        "dry_run": bool(dry_run),
        "window_seconds": int(window_seconds),
        "start_emotion": str(start_emotion),
        "num_seed_runs": len(events),
        "num_groups": len(groups_out),
        "groups": groups_out,
        "moved": moved,
        "cleaned": cleaned,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Group per-emotion similarity runs into bash-level run_id folders.")
    p.add_argument(
        "--root",
        default="auto_experiments/task_similarity/results/anger_pd_delta_similarity",
        help="Root results directory.",
    )
    p.add_argument("--window_seconds", type=int, default=10 * 60)
    p.add_argument(
        "--start_emotion",
        default="emotion",
        help="Emotion name that marks the start of a bash-level session (default: 'emotion').",
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    report = regroup(
        Path(args.root),
        window_seconds=int(args.window_seconds),
        start_emotion=str(args.start_emotion),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
