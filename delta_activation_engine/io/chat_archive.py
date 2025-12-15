"""
Responsible: delta_activation_engine/io/chat_archive.py
Purpose: Utilities to hash chat delta activation runs and archive duplicate folders.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Dict, List


def compute_delta_hash(run_dir: Path) -> str:
    """
    Compute a stable hash for a chat delta activation run based on its delta NPZ files.

    We hash the raw NPZ bytes for all files under `deltas/`, in lexicographic filename order.
    """
    deltas_dir = run_dir / "deltas"
    if not deltas_dir.is_dir():
        raise FileNotFoundError(f"Missing deltas directory for run: {run_dir}")

    delta_files = sorted(p for p in deltas_dir.iterdir() if p.is_file() and p.suffix == ".npz")
    if not delta_files:
        raise ValueError(f"No delta NPZ files found under: {deltas_dir}")

    hasher = hashlib.sha256()
    for path in delta_files:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def group_runs_by_hash(root: Path) -> Dict[str, List[Path]]:
    """
    Group run directories under `root` by their delta hash.

    Only directories containing a `deltas/` subdirectory with NPZ files are considered.
    """
    groups: Dict[str, List[Path]] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (run_dir / "deltas").is_dir():
            continue
        try:
            h = compute_delta_hash(run_dir)
        except (FileNotFoundError, ValueError):
            # Skip incomplete runs that lack usable delta files.
            continue
        groups.setdefault(h, []).append(run_dir)
    return groups


def archive_duplicates(root: Path) -> Dict[str, List[Path]]:
    """
    Move duplicate runs into `root / "archive"` based on delta hash equality.

    For each hash group, the lexicographically smallest run directory name is kept in-place
    as the canonical copy; all others are moved into the archive directory.
    Returns a mapping from hash -> list of archived run paths.
    """
    groups = group_runs_by_hash(root)
    archive_root = root / "archive"
    archive_root.mkdir(exist_ok=True)

    archived: Dict[str, List[Path]] = {}
    for h, runs in groups.items():
        if len(runs) <= 1:
            continue
        ordered = sorted(runs, key=lambda p: p.name)
        canonical = ordered[0]
        duplicates = ordered[1:]

        moved_paths: List[Path] = []
        for dup in duplicates:
            dest = archive_root / dup.name
            shutil.move(str(dup), str(dest))
            moved_paths.append(dest)

        if moved_paths:
            archived[h] = moved_paths

    return archived
