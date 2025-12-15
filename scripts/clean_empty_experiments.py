"""
Utility to remove empty experiment directories under a given root (default:
results/bfcl_qwen3/live). A directory is considered empty if it contains no
files anywhere in its subtree. The root directory itself is never removed.

Usage:
  python -m scripts.clean_empty_experiments [ROOT_DIR]
  # or
  python scripts/clean_empty_experiments.py [ROOT_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Iterable, List


def _iter_dirs_bottom_up(root: Path) -> Iterable[Path]:
    """Yield all directories under root, deepest first. Excludes root itself."""
    # Collect then sort by depth (parts length) descending so children come first
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    return dirs


def clean_empty_experiments(root: Path | str) -> List[Path]:
    """Remove recursively-empty directories under root and return removed paths.

    A directory is removed if and only if it contains no files anywhere in its
    subtree. Only directories under `root` are considered; `root` itself is not
    removed even if empty.
    """
    root_path = Path(root)
    removed: List[Path] = []

    if not root_path.exists() or not root_path.is_dir():
        return removed

    # Bottom-up pass: remove directories that are empty. Removing deepest empty
    # dirs first will make higher-level dirs become empty in the same pass.
    for d in _iter_dirs_bottom_up(root_path):
        try:
            # If directory has no entries, it is empty and safe to remove.
            # We do a fast check: empty iterator means no files or subdirs.
            next(d.iterdir())
        except StopIteration:
            shutil.rmtree(d)
            removed.append(d)
        except FileNotFoundError:
            # Directory may have been removed as part of an earlier rmtree.
            continue

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove empty experiment directories under ROOT_DIR.")
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=str(Path("results") / "bfcl_qwen3" / "live"),
        help="Root directory to clean (default: results/bfcl_qwen3/live)",
    )
    args = parser.parse_args()

    removed = clean_empty_experiments(args.root_dir)
    if removed:
        print("Removed:")
        for p in removed:
            # Print relative to root for easier reading when called by hand
            try:
                rel = Path(args.root_dir).resolve().relative_to(Path.cwd().resolve())
            except Exception:
                rel = Path(args.root_dir)
            try:
                rp = p.resolve().relative_to(rel.resolve())  # type: ignore[attr-defined]
            except Exception:
                rp = p
            print(f"- {rp}")
    else:
        print("No empty experiment directories found.")


if __name__ == "__main__":
    main()

