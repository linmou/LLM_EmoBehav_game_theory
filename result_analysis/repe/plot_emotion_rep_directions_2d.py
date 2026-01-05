from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import yaml
from transformers import AutoConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from neuro_manipulation.rep_reader_viz import (
    collect_direction_points,
    reduce_vectors_to_2d,
    emotion_reader_cache_path,
    infer_repe_config_for_model,
)


def _model_id(model_path: str) -> str:
    p = Path(model_path)
    if p.name:
        return p.name.rstrip("/")
    return str(model_path).rstrip("/").split("/")[-1]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _num_hidden_layers(model_path: str) -> int:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    candidates = [cfg]
    for nested in ("text_config", "language_config"):
        nested_cfg = getattr(cfg, nested, None)
        if nested_cfg is not None:
            candidates.append(nested_cfg)

    for obj in candidates:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    raise ValueError(f"Cannot determine num hidden layers for {model_path} (config={type(cfg)})")


def _plot_and_save_with_coords(
    coords, meta, out_png: Path, out_jsonl: Path, title: str, method_label: str
) -> None:
    emotions = sorted({m["emotion"] for m in meta})

    fig, ax = plt.subplots(figsize=(9, 7))
    for emo in emotions:
        idx = [i for i, m in enumerate(meta) if m["emotion"] == emo]
        xs = coords[idx, 0]
        ys = coords[idx, 1]
        ax.scatter(xs, ys, s=14, alpha=0.85, label=emo)

    ax.set_title(title)
    ax.set_xlabel(f"{method_label}1")
    ax.set_ylabel(f"{method_label}2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for m, (x, y) in zip(meta, coords.tolist()):
            rec = dict(m)
            rec["x"] = float(x)
            rec["y"] = float(y)
            f.write(json.dumps(rec) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot cached emotion RepReader layer direction vectors in 2D (PCA)."
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("neuro_manipulation/representation_storage"),
        help="Directory containing emotion_rep_reader_*.pkl files.",
    )
    ap.add_argument(
        "--fallback-scan",
        action="store_true",
        help="If the exact hash-derived cache file is missing, scan storage-dir and use the newest cache for that model.",
    )
    ap.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only process models whose path contains this substring.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix added to output filenames so you can run multiple settings without overwriting.",
    )
    ap.add_argument(
        "--method",
        choices=["pca", "tsne", "umap"],
        default="pca",
        help="2D reduction method applied to stored direction vectors.",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for tsne/umap (common: euclidean, cosine).",
    )
    ap.add_argument(
        "--normalize",
        choices=["none", "l2"],
        default="none",
        help="Optional pre-normalization of direction vectors before 2D reduction.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as --storage-dir).",
    )
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = args.storage_dir

    cfg = _load_yaml(args.config)
    model_paths = cfg.get("models")
    if not isinstance(model_paths, list) or not model_paths:
        raise ValueError(f"No models list found in {args.config}")

    for model_path in model_paths:
        if args.model_filter and args.model_filter not in model_path:
            continue
        pkl_path = None
        repe_cfg = infer_repe_config_for_model(model_path, cfg)
        hidden_layers = list(range(-1, -_num_hidden_layers(model_path) - 1, -1))
        hashed = emotion_reader_cache_path(repe_cfg, hidden_layers=hidden_layers)
        print(
            "[hash] "
            f"{model_path} -> {hashed.name} "
            f"(data_dir={repe_cfg.get('data_dir')}, seed={repe_cfg.get('emotion_data_seed', 0)}, "
            f"method={repe_cfg.get('direction_method')}, rep_token={repe_cfg.get('rep_token')}, "
            f"n_difference={repe_cfg.get('n_difference')}, layers={len(hidden_layers)})"
        )
        if hashed.exists():
            pkl_path = hashed

        if pkl_path is None:
            if not args.fallback_scan:
                print(f"[skip] hash-derived cache missing (use --fallback-scan to search): {hashed}")
                continue
            matches = []
            for cand in args.storage_dir.glob("emotion_rep_reader_*.pkl"):
                try:
                    with cand.open("rb") as f:
                        obj = pickle.load(f)
                except Exception:
                    continue
                cfg_args = obj.get("args") if isinstance(obj, dict) else None
                if isinstance(cfg_args, dict) and cfg_args.get("model_name_or_path") == model_path:
                    matches.append(cand)

            if not matches:
                print(f"[skip] no cached emotion reader for {model_path} under {args.storage_dir}")
                continue
            pkl_path = max(matches, key=lambda p: p.stat().st_mtime)
            print(f"[warn] hash-miss; using newest match: {model_path} -> {pkl_path.name}")

        print(f"[use] {model_path} -> {pkl_path}")
        with pkl_path.open("rb") as f:
            emotion_rep_readers = pickle.load(f)

        model_name = _model_id(model_path)
        vectors, meta = collect_direction_points(emotion_rep_readers, model_id=model_name)
        out_png = args.out_dir / f"{model_name}_emotion_rep_directions_pca2d.png"
        out_jsonl = args.out_dir / f"{model_name}_emotion_rep_directions_pca2d.jsonl"
        coords = reduce_vectors_to_2d(
            vectors,
            method=args.method,
            seed=args.seed,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            normalize=args.normalize,
        )
        out_png = out_png.with_name(out_png.name.replace("_pca2d", f"_{args.method}2d"))
        out_jsonl = out_jsonl.with_name(out_jsonl.name.replace("_pca2d", f"_{args.method}2d"))
        if args.tag:
            out_png = out_png.with_name(out_png.stem + f"_{args.tag}" + out_png.suffix)
            out_jsonl = out_jsonl.with_name(out_jsonl.stem + f"_{args.tag}" + out_jsonl.suffix)
        method_label = {"pca": "PC", "tsne": "tSNE", "umap": "UMAP"}[args.method]
        _plot_and_save_with_coords(
            coords,
            meta,
            out_png=out_png,
            out_jsonl=out_jsonl,
            title=f"{model_name}: emotion direction vectors ({args.method})",
            method_label=method_label,
        )
        print(f"[ok] {model_name}: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
