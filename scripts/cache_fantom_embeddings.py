#!/usr/bin/env python3
"""Cache FANToM belief-generation embeddings for one file or a directory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.datasets.fantom import FantomDataset


@dataclass
class CacheResult:
    file_path: Path
    keys_path: Path
    vecs_path: Path
    num_keys: int
    dimension: int


def collect_strings(ds: FantomDataset) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in ds.items:
        if isinstance(item.ground_truth, str):
            gt = item.ground_truth.strip()
            if gt and gt not in seen:
                seen.add(gt)
                out.append(gt)
        metadata = item.metadata or {}
        wrong = metadata.get("wrong_answer")
        if isinstance(wrong, str):
            wa = wrong.strip()
            if wa and wa not in seen:
                seen.add(wa)
                out.append(wa)
    return out


def infer_task_type_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("fantom_"):
        return stem[len("fantom_"):]
    return stem


def _build_config(data_path: Path, task_type: str, limit: int) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="fantom",
        task_type=task_type,
        data_path=data_path,
        base_data_dir=None,
        sample_limit=(None if limit <= 0 else limit),
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def _persist_embeddings(keys: Sequence[str], vecs: np.ndarray, prefix: Path) -> Tuple[Path, Path]:
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    keys_path = Path(str(base) + "_keys.json")
    vecs_path = Path(str(base) + "_vecs.npy")

    with keys_path.open("w", encoding="utf-8") as f:
        json.dump(list(keys), f, ensure_ascii=False, indent=2)
    np.save(vecs_path, vecs.astype(np.float32))

    return keys_path, vecs_path


def cache_embeddings_for_file(
    data_path: Path,
    task_type: Optional[str],
    *,
    limit: int = 0,
    persist_prefix: Optional[Path] = None,
) -> CacheResult:
    data_path = data_path.resolve()
    task = task_type or infer_task_type_from_filename(data_path)
    cfg = _build_config(data_path, task, limit)

    ds = FantomDataset(cfg, prompt_wrapper=None)
    keys = collect_strings(ds)

    print(f"Unique strings to embed ({data_path.name}): {len(keys)}")
    print(f"Exceeds LRU capacity (2048)? {'YES' if len(keys) > 2048 else 'NO'}")

    embedder = ds._get_embedder()
    vecs = ds._encode_texts_cached(embedder, keys, True)

    if persist_prefix is None:
        persist_prefix = data_path.with_name(data_path.stem + "_emb")
    else:
        persist_prefix = Path(persist_prefix)

    keys_path, vecs_path = _persist_embeddings(keys, vecs, persist_prefix)

    approx_mb = (vecs.size * 4) / (1024 * 1024)
    print(
        f"Saved {len(keys)} keys -> {keys_path.name}, {vecs_path.name}. "
        f"dim={vecs.shape[1] if vecs.size else 0}, approx {approx_mb:.2f} MB"
    )
    return CacheResult(
        file_path=data_path,
        keys_path=keys_path,
        vecs_path=vecs_path,
        num_keys=len(keys),
        dimension=int(vecs.shape[1]) if vecs.size else 0,
    )


def cache_embeddings_in_dir(
    data_dir: Path,
    *,
    limit: int = 0,
    match_substring: str = "gen",
) -> List[CacheResult]:
    data_dir = data_dir.resolve()
    files = sorted(p for p in data_dir.glob("*.jsonl") if match_substring in p.name)
    if not files:
        print(f"No *{match_substring}*.jsonl files found in {data_dir}")
        return []

    results: List[CacheResult] = []
    for file_path in files:
        try:
            result = cache_embeddings_for_file(file_path, None, limit=limit, persist_prefix=None)
            results.append(result)
        except Exception as exc:  # pragma: no cover - surfaced to CLI users
            print(f"ERROR processing {file_path.name}: {exc}")
    if results:
        total_keys = sum(r.num_keys for r in results)
        dim = results[0].dimension if results[0].dimension else 0
        print(
            f"\nDone. Cached {len(results)} files. Total unique keys (non-dedup sum): {total_keys}."
        )
        if dim:
            approx_mb = total_keys * dim * 4 / (1024 * 1024)
            print(f"If held in-memory together, approx RAM: {approx_mb:.2f} MB at dim={dim}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_path", type=Path, help="Path to a single FANToM JSONL file")
    ap.add_argument("--task_type", type=str, help="Explicit task type for --data_path")
    ap.add_argument("--data_dir", type=Path, help="Directory containing FANToM JSONL files")
    ap.add_argument("--match", type=str, default="gen", help="Substring filter when using --data_dir")
    ap.add_argument("--persist", type=Path, help="Custom output prefix for --data_path runs")
    ap.add_argument("--limit", type=int, default=0, help="Optional sample limit (0 = all)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if bool(args.data_path) == bool(args.data_dir):
        ap.error("Exactly one of --data_path or --data_dir must be supplied.")

    if args.data_path:
        cache_embeddings_for_file(
            args.data_path,
            args.task_type,
            limit=args.limit,
            persist_prefix=args.persist,
        )
        return 0

    cache_embeddings_in_dir(
        args.data_dir,
        limit=args.limit,
        match_substring=args.match,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
