"""
Responsible: auto_experiments/task_similarity/run_pd_probe_transfer.py
Purpose: Run the PD-probe transfer analysis specified in PD_PROBE_TRANSFER_SPEC.md.

Consumes an existing emotion_pd_delta_similarity run directory layout:
  <run_dir>/<model>/<emotion>/seed_<seed>/{delta_*.npy, metadata.json}

Writes CSVs:
  - pd_probe_transfer_auc_by_layer.csv
  - pd_probe_transfer_controlled_layer_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .pd_probe_transfer import evaluate_transfer_auc_by_layer, train_pd_probes_per_layer


def _load_split_indices(path: Path, *, split: str) -> List[int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = str(split).strip().lower()
    if split == "train":
        return [int(i) for i in manifest.get("train_indices", [])]
    if split == "test":
        return [int(i) for i in manifest.get("test_indices", [])]
    raise ValueError(f"unsupported split: {split!r}")


def _read_chosen_behavior_map(detailed_results_csv: Path) -> Dict[Tuple[str, float, int], int]:
    """
    Returns mapping (emotion, intensity, item_id) -> defect(1)/cooperate(0).
    Rows with missing chosen_behavior are skipped.
    """
    import pandas as pd

    df = pd.read_csv(detailed_results_csv)
    need = {"emotion", "intensity", "item_id", "chosen_behavior"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"detailed_results.csv missing columns: {sorted(need - set(df.columns))}")

    out: Dict[Tuple[str, float, int], int] = {}
    for row in df.itertuples(index=False):
        emo = str(getattr(row, "emotion"))
        intensity = float(getattr(row, "intensity"))
        item_id = int(getattr(row, "item_id"))
        beh = getattr(row, "chosen_behavior")
        if beh is None or (isinstance(beh, float) and np.isnan(beh)):
            continue
        beh_s = str(beh).strip().lower()
        if beh_s not in {"defect", "cooperate"}:
            continue
        out[(emo, float(intensity), int(item_id))] = 1 if beh_s == "defect" else 0
    return out


def _bootstrap_auc_ci(
    *,
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    y2 = np.asarray(y, dtype=np.int64)
    s2 = np.asarray(scores, dtype=np.float32)
    if y2.shape != s2.shape:
        raise ValueError("y and scores must have the same shape")
    if y2.size < 2 or len(np.unique(y2)) < 2:
        return float("nan"), float("nan")

    n = int(y2.size)
    aucs: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yy = y2[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(float(roc_auc_score(yy, s2[idx])))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.array(aucs, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


@dataclass(frozen=True)
class _RunContext:
    model_dir: Path
    seed: str
    intensities: List[float]
    controlled_layers: List[int]
    item_ids: List[int]  # tensor-order item ids
    train_item_indices: List[int]  # tensor indices
    test_item_ids: List[int]  # item ids
    test_item_indices: List[int]  # tensor indices aligned with test_item_ids
    results_dir: Path
    label_map: Dict[Tuple[str, float, int], int]


def _discover_run_context(run_dir: Path) -> _RunContext:
    run_dir = Path(run_dir)
    def _looks_like_model_dir(p: Path) -> bool:
        if not p.is_dir():
            return False
        for emo in p.iterdir():
            if not emo.is_dir():
                continue
            for sd in emo.iterdir():
                if not (sd.is_dir() and sd.name.startswith("seed_")):
                    continue
                if (sd / "metadata.json").exists():
                    return True
        return False

    model_dirs = sorted([p for p in run_dir.iterdir() if _looks_like_model_dir(p)])
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found under {run_dir}")
    if len(model_dirs) > 1:
        # KISS: require the user to point at a single-model run dir.
        raise ValueError(f"Multiple model dirs under {run_dir}; expected one: {[p.name for p in model_dirs]}")
    model_dir = model_dirs[0]

    emotion_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
    if not emotion_dirs:
        raise FileNotFoundError(f"No emotion directories under {model_dir}")

    # Pick the first seed_* folder as the reference.
    ref_seed_dirs = sorted([p for p in emotion_dirs[0].iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not ref_seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs under {emotion_dirs[0]}")
    ref_seed_dir = ref_seed_dirs[0]
    seed = ref_seed_dir.name

    meta = json.loads((ref_seed_dir / "metadata.json").read_text(encoding="utf-8"))
    intensities = [float(x) for x in meta["intensities"]]
    controlled_layers = [int(x) for x in meta["controlled_layers"]]
    item_ids = [int(x) for x in meta.get("item_ids", [])]
    if not item_ids:
        # Fall back to implicit 0..n-1 if not stored.
        n_samples = int(meta.get("n_samples", 0))
        if n_samples > 0:
            item_ids = list(range(n_samples))
        else:
            # Last resort: infer from delta tensor header.
            delta_pd_path = ref_seed_dir / "delta_pd.npy"
            if not delta_pd_path.exists():
                raise ValueError("metadata.json missing item_ids/n_samples and delta_pd.npy not found")
            arr = np.load(delta_pd_path, mmap_mode="r")
            if arr.ndim != 4:
                raise ValueError(f"delta_pd.npy must be 4D, got {arr.shape}")
            item_ids = list(range(int(arr.shape[1])))
    id_to_idx = {int(item_id): idx for idx, item_id in enumerate(item_ids)}

    split_manifest = Path(meta["split_manifest"])
    train_ids = [int(i) for i in _load_split_indices(split_manifest, split="train") if int(i) in id_to_idx]
    test_ids = [int(i) for i in _load_split_indices(split_manifest, split="test") if int(i) in id_to_idx]
    train_item_indices = [int(id_to_idx[i]) for i in train_ids]
    test_item_indices = [int(id_to_idx[i]) for i in test_ids]

    raw_results_path = Path(meta["raw_results_path"])
    # Historical metadata sometimes stores the raw_results.json *file* path, not its directory.
    results_dir = raw_results_path.parent if raw_results_path.suffix == ".json" else raw_results_path
    detailed = results_dir / "detailed_results.csv"
    if not detailed.exists():
        raise FileNotFoundError(f"Missing detailed_results.csv: {detailed}")
    label_map = _read_chosen_behavior_map(detailed)

    return _RunContext(
        model_dir=model_dir,
        seed=seed,
        intensities=intensities,
        controlled_layers=controlled_layers,
        item_ids=item_ids,
        train_item_indices=train_item_indices,
        test_item_ids=test_ids,
        test_item_indices=test_item_indices,
        results_dir=results_dir,
        label_map=label_map,
    )


def run_pd_probe_transfer(*, run_dir: Path, out_dir: Path, n_boot: int = 1000, random_seed: int = 0) -> None:
    ctx = _discover_run_context(Path(run_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emotion_dirs = sorted([p for p in ctx.model_dir.iterdir() if p.is_dir()])
    emotions = [p.name for p in emotion_dirs]

    # Reference PD deltas (train once per run/seed).
    ref_seed_dir = ctx.model_dir / emotions[0] / ctx.seed
    delta_pd = np.load(ref_seed_dir / "delta_pd.npy", mmap_mode="r")
    delta_pd_coop = np.load(ref_seed_dir / "delta_pd_cooperate.npy", mmap_mode="r")
    probes = train_pd_probes_per_layer(
        delta_pd=delta_pd,
        delta_pd_cooperate=delta_pd_coop,
        train_item_ids=ctx.train_item_indices,
        l2_normalize=True,
    )

    rng = np.random.default_rng(int(random_seed))
    int_to_idx = {float(v): i for i, v in enumerate(ctx.intensities)}

    by_layer_path = out_dir / "pd_probe_transfer_auc_by_layer.csv"
    summary_path = out_dir / "pd_probe_transfer_controlled_layer_summary.csv"

    with by_layer_path.open("w", newline="", encoding="utf-8") as f_layer, summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as f_sum:
        w_layer = csv.writer(f_layer)
        w_sum = csv.writer(f_sum)
        w_layer.writerow(["emotion", "intensity", "layer", "auc", "ci_low", "ci_high", "n_test_items"])
        w_sum.writerow(
            [
                "emotion",
                "intensity",
                "defect_rate",
                "n_test_items",
                "mean_auc_controlled",
                "mean_ci_low",
                "mean_ci_high",
                "median_auc_controlled",
                "median_ci_low",
                "median_ci_high",
                "controlled_layers",
            ]
        )

        controlled = [int(x) for x in ctx.controlled_layers]

        for emo_dir in emotion_dirs:
            emo = emo_dir.name
            seed_dir = emo_dir / ctx.seed
            if not seed_dir.exists():
                continue
            delta_emotion = np.load(seed_dir / "delta_emotion.npy", mmap_mode="r")

            for intensity in ctx.intensities:
                i_int = int_to_idx[float(intensity)]

                # Build labels aligned to test_items, dropping items without a label.
                items: List[int] = []
                item_ids: List[int] = []
                y_list: List[int] = []
                for item_id, item_idx in zip(ctx.test_item_ids, ctx.test_item_indices):
                    key = (emo, float(intensity), int(item_id))
                    if key not in ctx.label_map:
                        continue
                    item_ids.append(int(item_id))
                    items.append(int(item_idx))
                    y_list.append(int(ctx.label_map[key]))
                if not items:
                    continue
                y = np.asarray(y_list, dtype=np.int64)

                aucs = evaluate_transfer_auc_by_layer(
                    probes=probes,
                    delta_emotion=delta_emotion,
                    test_item_ids=items,
                    y_defect=y,
                    intensity_index=i_int,
                    l2_normalize=True,
                )

                # Per-layer CI via bootstrapping scores.
                for layer, auc in enumerate(aucs.tolist()):
                    x = np.asarray(delta_emotion[i_int, items, layer, :], dtype=np.float32)
                    # score normalization is handled inside evaluate_transfer_auc_by_layer; do it again for CI.
                    nrm = np.linalg.norm(x, axis=1, keepdims=True) + np.float32(1e-12)
                    scores = probes[layer].score(x / nrm)
                    ci_lo, ci_hi = _bootstrap_auc_ci(y=y, scores=scores, n_boot=n_boot, rng=rng)
                    w_layer.writerow([emo, float(intensity), int(layer), float(auc), float(ci_lo), float(ci_hi), len(items)])

                # Controlled-layer summaries.
                ctrl_aucs = np.asarray([aucs[lyr] for lyr in controlled if 0 <= int(lyr) < len(aucs)], dtype=np.float32)
                finite_ctrl = ctrl_aucs[np.isfinite(ctrl_aucs)]
                mean_auc = float(np.mean(finite_ctrl)) if finite_ctrl.size else float("nan")
                median_auc = float(np.median(finite_ctrl)) if finite_ctrl.size else float("nan")

                # Bootstrap mean/median by resampling items and recomputing per-layer AUCs.
                mean_boot: List[float] = []
                median_boot: List[float] = []
                if y.size >= 2 and len(np.unique(y)) >= 2 and ctrl_aucs.size:
                    from sklearn.metrics import roc_auc_score

                    for _ in range(int(n_boot)):
                        idx = rng.integers(0, y.size, size=y.size)
                        yy = y[idx]
                        if len(np.unique(yy)) < 2:
                            continue
                        per_layer: List[float] = []
                        for lyr in controlled:
                            x = np.asarray(
                                delta_emotion[i_int, np.asarray(items, dtype=np.int64)[idx], int(lyr), :],
                                dtype=np.float32,
                            )
                            nrm = np.linalg.norm(x, axis=1, keepdims=True) + np.float32(1e-12)
                            s = probes[int(lyr)].score(x / nrm)
                            per_layer.append(float(roc_auc_score(yy, s)))
                        arr = np.asarray(per_layer, dtype=np.float64)
                        mean_boot.append(float(np.mean(arr)))
                        median_boot.append(float(np.median(arr)))

                if mean_boot:
                    m_lo, m_hi = np.percentile(np.asarray(mean_boot, dtype=np.float64), [2.5, 97.5])
                else:
                    m_lo, m_hi = float("nan"), float("nan")
                if median_boot:
                    md_lo, md_hi = np.percentile(np.asarray(median_boot, dtype=np.float64), [2.5, 97.5])
                else:
                    md_lo, md_hi = float("nan"), float("nan")

                w_sum.writerow(
                    [
                        emo,
                        float(intensity),
                        float(np.mean(y)),
                        int(y.size),
                        float(mean_auc),
                        float(m_lo),
                        float(m_hi),
                        float(median_auc),
                        float(md_lo),
                        float(md_hi),
                        json.dumps(controlled),
                    ]
                )


def main() -> None:
    p = argparse.ArgumentParser(description="Run per-layer PD-probe transfer AUC analysis on a delta-similarity run.")
    p.add_argument("--run_dir", required=True, help="Path to <run_id> directory produced by emotion_pd_delta_similarity.")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <run_dir>/pd_probe_transfer/).",
    )
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--random_seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "pd_probe_transfer")
    run_pd_probe_transfer(run_dir=run_dir, out_dir=out_dir, n_boot=int(args.n_boot), random_seed=int(args.random_seed))


if __name__ == "__main__":
    main()
