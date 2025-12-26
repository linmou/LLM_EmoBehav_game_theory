"""
Responsible: delta_activation_engine/io/chat_archive.py
Purpose: Verify chat delta activation runs are grouped and archived by hash of their delta NPZ files.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from delta_activation_engine.io.chat_archive import (  # type: ignore[attr-defined]
    archive_duplicates,
    compute_delta_hash,
    group_runs_by_hash,
)


def _make_run(root: Path, name: str, vectors: dict[str, np.ndarray]) -> Path:
    run = root / name
    deltas = run / "deltas"
    deltas.mkdir(parents=True, exist_ok=True)
    # baseline/metadata are irrelevant for hashing but we mirror real layout a bit
    (run / "baseline.npz").write_bytes(b"baseline")
    (run / "metadata.json").write_text("{}", encoding="utf-8")
    for fname, vec in vectors.items():
        path = deltas / fname
        np.savez_compressed(path, vector=vec)
    return run


def test_compute_delta_hash_ignores_file_order(tmp_path: Path) -> None:
    # Same two vectors, different file ordering -> same hash
    root = tmp_path / "chat"
    root.mkdir()

    v1 = np.array([1.0, 2.0], dtype=np.float32)
    v2 = np.array([3.0, 4.0], dtype=np.float32)

    run_a = _make_run(
        root,
        "run_a",
        {
            "emotion=anger_int=0.0.npz": v1,
            "emotion=anger_int=1.0.npz": v2,
        },
    )
    run_b = _make_run(
        root,
        "run_b",
        {
            "emotion=anger_int=1.0.npz": v2,
            "emotion=anger_int=0.0.npz": v1,
        },
    )

    h_a = compute_delta_hash(run_a)
    h_b = compute_delta_hash(run_b)
    assert h_a == h_b


def test_group_runs_by_hash_finds_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "chat"
    root.mkdir()

    shared = np.array([0.1, 0.2], dtype=np.float32)
    unique = np.array([5.0, 6.0], dtype=np.float32)

    run1 = _make_run(root, "run1", {"e0.npz": shared})
    run2 = _make_run(root, "run2", {"e0.npz": shared})
    _make_run(root, "run3", {"e0.npz": unique})

    # Incomplete run with empty deltas directory should be ignored, not crash
    empty = root / "run_empty"
    (empty / "deltas").mkdir(parents=True, exist_ok=True)

    groups = group_runs_by_hash(root)
    # At least one hash should map to the two shared runs
    duplicate_groups = [runs for runs in groups.values() if set(runs) == {run1, run2}]
    assert len(duplicate_groups) == 1


def test_archive_duplicates_moves_non_canonical_runs(tmp_path: Path) -> None:
    root = tmp_path / "chat"
    root.mkdir()

    shared = np.array([7.0, 8.0], dtype=np.float32)
    other = np.array([9.0, 10.0], dtype=np.float32)

    # Lexicographically smallest should be canonical
    run_a = _make_run(root, "run_a", {"e0.npz": shared})
    run_b = _make_run(root, "run_b", {"e0.npz": shared})
    run_c = _make_run(root, "run_c", {"e0.npz": other})

    archived = archive_duplicates(root)

    # Only run_b should be archived; run_a is canonical, run_c is unique
    archive_root = root / "archive"
    assert (root / run_a.name).is_dir()
    assert not (root / run_b.name).exists()
    assert (archive_root / run_b.name).is_dir()
    assert (root / run_c.name).is_dir()

    # Function should report archived runs keyed by their hash
    all_archived = {p.name for runs in archived.values() for p in runs}
    assert all_archived == {run_b.name}
