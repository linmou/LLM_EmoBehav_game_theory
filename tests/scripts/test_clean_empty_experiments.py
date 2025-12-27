"""
Test file for scripts/clean_empty_experiments.py

Purpose: Verify that the cleaner removes only directories under a given root
that contain no files anywhere (recursively), and never deletes the root.
"""

import os
from pathlib import Path

import importlib.util


def make_file(p: Path, content: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_clean_empty_experiments_removes_only_recursively_empty_dirs(tmp_path: Path) -> None:
    # Arrange: create a fake results tree under tmp_path/live
    root = tmp_path / "live"
    root.mkdir(parents=True)

    # Empty experiment dirs
    (root / "exp_empty_top").mkdir()
    (root / "exp_empty_nested" / "only_dirs" / "deeper").mkdir(parents=True)

    # Non-empty experiment (has a file nested)
    make_file(root / "exp_with_file" / "sub" / "out.json", "{}")

    # Non-empty: file directly under experiment dir
    make_file(root / "exp_with_direct_file" / "log.txt", "log")

    # Non-empty: file directly under root (should not be deleted and should not block others)
    make_file(root / "root_note.txt", "note")

    # Act: import and run the cleaner on our temp root (load by path to avoid
    # conflicts with any site-packages module named "scripts")
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "scripts" / "clean_empty_experiments.py"
    spec = importlib.util.spec_from_file_location("clean_empty_experiments", mod_path)
    assert spec and spec.loader, "Failed to build import spec for cleaner module"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    removed = set(mod.clean_empty_experiments(root))

    # Assert: only the recursively empty experiment dirs are removed
    assert (root / "exp_empty_top").exists() is False
    assert (root / "exp_empty_nested").exists() is False

    # Non-empty experiment dirs must remain
    assert (root / "exp_with_file").is_dir()
    assert (root / "exp_with_direct_file").is_dir()

    # Root must remain and file under root must remain
    assert root.is_dir()
    assert (root / "root_note.txt").is_file()

    # The set of removed directories should include both empty ones (absolute paths)
    expected_removed = {root / "exp_empty_top", root / "exp_empty_nested", root / "exp_empty_nested" / "only_dirs", root / "exp_empty_nested" / "only_dirs" / "deeper"}
    # removed should at least be a superset of expected_removed (exact content may include intermediate dirs)
    assert expected_removed.issubset(removed)
