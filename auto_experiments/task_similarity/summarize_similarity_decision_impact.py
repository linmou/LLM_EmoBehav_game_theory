"""
Responsible: auto_experiments/task_similarity/summarize_similarity_decision_impact.py
Purpose: Summarize how cosine(Δ^anger, Δ^pd) relates to PD decisions.

Inputs: a decision-impact directory produced by
`auto_experiments.task_similarity.analyze_similarity_decision_impact`, containing:
- layer_impact_summary.csv
- samples_with_decision.csv

Outputs:
- Prints concise tables to stdout
- Optionally writes CSV slices to an output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import numpy as np


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same shape")
    if x.size < 3:
        return float("nan")
    vx = float(np.var(x))
    vy = float(np.var(y))
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def perm_p_value_pearson_abs(x: np.ndarray, y01: np.ndarray, *, B: int = 20000, seed: int = 0) -> float:
    """
    Two-sided permutation p-value for |pearson_r(x,y)|, shuffling y.
    Vectorized in blocks so it stays fast for n~1k and B~20k.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y01, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same shape")
    if int(B) <= 0:
        raise ValueError("B must be positive")

    r_obs = pearson_r(x, y)
    if not np.isfinite(r_obs):
        return float("nan")

    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom == 0.0:
        return float("nan")

    rng = np.random.default_rng(int(seed))
    n = int(x0.size)
    block = 2000
    exceed = 0
    done = 0
    while done < int(B):
        b = min(block, int(B) - done)
        idx = np.empty((b, n), dtype=np.int64)
        for i in range(b):
            idx[i, :] = rng.permutation(n)
        dots = (y0[idx] @ x0) / denom
        exceed += int(np.sum(np.abs(dots) >= abs(float(r_obs)) - 1e-12))
        done += b
    return float((exceed + 1) / (int(B) + 1))


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values in the same order as p_values.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    if p.size == 0:
        return []
    m = int(p.size)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        k = m - rank + 1
        val = float(p[idx]) * m / k
        if val > prev:
            val = prev
        prev = val
        q[idx] = val
    q = np.clip(q, 0.0, 1.0)
    return [float(v) for v in q]


def _last_layer_index(joined_rows: pd.DataFrame) -> int:
    if "layer" not in joined_rows.columns:
        raise ValueError("missing column: layer")
    return int(joined_rows["layer"].astype(int).max())


def _corr_and_perm_by_intensity(
    df: pd.DataFrame, *, B: int, seed: int, label: str
) -> pd.DataFrame:
    required = {"intensity", "cosine", "defect"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out_rows = []
    for intensity in sorted(df["intensity"].unique()):
        sub = df[df["intensity"] == intensity].dropna(subset=["cosine", "defect"])
        x = sub["cosine"].astype(float).to_numpy()
        y = sub["defect"].astype(int).to_numpy()
        # Drop NaNs in x explicitly (pandas dropna might not catch non-float).
        mask = np.isfinite(x)
        x = x[mask]
        y = y[mask]
        r = pearson_r(x, y.astype(np.float64))
        p_perm = perm_p_value_pearson_abs(x, y.astype(np.float64), B=int(B), seed=int(seed))
        out_rows.append(
            {
                "metric": str(label),
                "intensity": float(intensity),
                "n": int(x.size),
                "r": float(r),
                "p_perm": float(p_perm),
                "perm_B": int(B),
                "perm_seed": int(seed),
            }
        )
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["q_bh_perm"] = bh_fdr(out["p_perm"].astype(float).tolist())
    return out


def compute_last_layer_significance(impact_dir: Path, *, B: int = 20000, seed: int = 0) -> pd.DataFrame:
    joined = _read_csv(Path(impact_dir) / "joined_rows.csv")
    last_layer = _last_layer_index(joined)
    sub = joined[joined["layer"].astype(int) == int(last_layer)].copy()
    out = _corr_and_perm_by_intensity(sub, B=int(B), seed=int(seed), label=f"last_layer_{last_layer}")
    if not out.empty:
        out["layer"] = int(last_layer)
    return out


def compute_last5_mean_significance(impact_dir: Path, *, B: int = 20000, seed: int = 0) -> pd.DataFrame:
    joined = _read_csv(Path(impact_dir) / "joined_rows.csv")
    last_layer = _last_layer_index(joined)
    last5 = list(range(int(last_layer) - 4, int(last_layer) + 1))
    sub = joined[joined["layer"].astype(int).isin(last5)].copy()
    sub = sub.dropna(subset=["cosine"])
    # Require all 5 layers present per (item_id,intensity) before averaging.
    g = sub.groupby(["item_id", "intensity"], as_index=False)
    agg = g.agg(
        mean_cosine=("cosine", "mean"),
        n_layers=("layer", "nunique"),
        defect=("defect", "first"),
    )
    agg = agg[agg["n_layers"].astype(int) == 5].copy()
    agg = agg.rename(columns={"mean_cosine": "cosine"})
    out = _corr_and_perm_by_intensity(agg, B=int(B), seed=int(seed), label=f"last5_mean_{last5[0]}_{last5[-1]}")
    if not out.empty:
        out["layers"] = f"{last5[0]}-{last5[-1]}"
    return out


def select_last_layers(layer_summary: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if "layer" not in layer_summary.columns:
        raise ValueError("missing column: layer")
    max_layer = int(layer_summary["layer"].max())
    last = set(range(max_layer - int(k) + 1, max_layer + 1))
    return layer_summary[layer_summary["layer"].astype(int).isin(last)].copy()


def top_abs_pearson_per_intensity(layer_summary: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    col = "pearson_r(defect,cosine)"
    if "intensity" not in layer_summary.columns:
        raise ValueError("missing column: intensity")
    if col not in layer_summary.columns:
        raise ValueError(f"missing column: {col}")

    df = layer_summary.copy()
    df = df.dropna(subset=[col])
    df["_abs_r"] = df[col].astype(float).abs()
    out = []
    for intensity in sorted(df["intensity"].unique()):
        sub = df[df["intensity"] == intensity].sort_values("_abs_r", ascending=False).head(int(top_k))
        out.append(sub)
    if not out:
        return df.iloc[0:0].drop(columns=["_abs_r"], errors="ignore")
    res = pd.concat(out, ignore_index=True)
    return res.drop(columns=["_abs_r"], errors="ignore")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize similarity→decision impact stats.")
    p.add_argument(
        "--impact_dir",
        required=True,
        help="Path like .../seed_20/decision_impact/anger",
    )
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--last_k", type=int, default=5)
    p.add_argument("--out_dir", default=None, help="Optional output dir to write CSV slices.")
    p.add_argument("--perm_B", type=int, default=20000, help="Permutation count for significance tests.")
    p.add_argument("--perm_seed", type=int, default=0, help="RNG seed for permutation tests.")
    args = p.parse_args()

    impact_dir = Path(args.impact_dir)
    layer_path = impact_dir / "layer_impact_summary.csv"
    samples_path = impact_dir / "samples_with_decision.csv"

    layer = _read_csv(layer_path)
    samples = _read_csv(samples_path)

    print(f"impact_dir: {impact_dir}")
    print(f"joined_samples: {len(samples)}")
    if "defect" in samples.columns and "intensity" in samples.columns:
        counts = samples.groupby(["intensity", "defect"]).size().rename("n").reset_index()
        print("\nDecision counts (defect=1 / cooperate=0):")
        print(counts.to_string(index=False))

    top = top_abs_pearson_per_intensity(layer, top_k=int(args.top_k))
    cols = [
        "intensity",
        "layer",
        "controlled",
        "n",
        "n_defect",
        "n_cooperate",
        "pearson_r(defect,cosine)",
        "mean_cos_defect",
        "mean_cos_cooperate",
        "mean_diff_defect_minus_coop",
    ]
    cols = [c for c in cols if c in top.columns]
    print(f"\nTop |pearson_r| per intensity (top_k={int(args.top_k)}):")
    print(top[cols].to_string(index=False))

    last = select_last_layers(layer, k=int(args.last_k))
    cols_last = [c for c in cols if c in last.columns]
    if not last.empty:
        last = last.sort_values(["intensity", "layer"])
    print(f"\nLast {int(args.last_k)} layers (by layer index):")
    print(last[cols_last].to_string(index=False))

    out_dir: Optional[Path] = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        top.to_csv(out_dir / "top_abs_pearson_per_intensity.csv", index=False)
        last.to_csv(out_dir / f"last_{int(args.last_k)}_layers.csv", index=False)

    # Significance: last layer cosine + last-5 mean cosine.
    sig_last = compute_last_layer_significance(impact_dir, B=int(args.perm_B), seed=int(args.perm_seed))
    sig_last5 = compute_last5_mean_significance(impact_dir, B=int(args.perm_B), seed=int(args.perm_seed))
    if not sig_last.empty:
        out_path = impact_dir / "last_layer_corr_significance.csv"
        sig_last.to_csv(out_path, index=False)
        print("\nSignificance: corr(last_layer_sim, defect) (two-sided permutation p):")
        print(sig_last.to_string(index=False))
    if not sig_last5.empty:
        out_path = impact_dir / "last5_mean_corr_significance.csv"
        sig_last5.to_csv(out_path, index=False)
        print("\nSignificance: corr(mean(last5_layers_sim), defect) (two-sided permutation p):")
        print(sig_last5.to_string(index=False))


if __name__ == "__main__":
    main()
