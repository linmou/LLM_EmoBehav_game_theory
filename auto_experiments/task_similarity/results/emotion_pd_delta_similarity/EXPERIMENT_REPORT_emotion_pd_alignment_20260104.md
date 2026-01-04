# Emotion → PD Decision Shift via Representation Alignment
Date: 2026-01-04

This report summarizes an exploration of whether **emotion interventions** (RepE-style) shift Prisoner’s Dilemma decisions, and how to demonstrate the mechanism using only the saved artifacts (hidden states / deltas / steering vectors / model decisions).

## Research hypotheses

1. **Behavioral**: Emotion interventions systematically change PD decisions:
   - **Anger → more defection**
   - **Happiness → more cooperation**
2. **Mechanistic**: The emotion-induced activation shift `delta_emotion` moves **along a PD “defect vs cooperate” direction** in hidden space. If we project `delta_emotion` onto a fixed PD steering axis, the projection should relate to observed defection rates.

## Runs analyzed (artifacts already on disk)

| Run ID | Model | Notes |
|---|---|---|
| `20260103_164613` | `Qwen2.5-3B-Instruct` | Best mechanistic alignment (projection ↔ defect rate positive) |
| `20260103_164620` | `Qwen2.5-1.5B-Instruct` | Mechanistic correlation weak |
| `20260103_175442` | `Qwen2.5-1.5B-Instruct` | Same summary numbers as `164620` in our exploration |
| `20260103_191226` | `Qwen2.5-0.5B-Instruct` | **Mechanistic correlation flips negative** (failure of the PD-axis explanation) |

All above runs have `config.json` containing:
- `result_dir`: path to `detailed_results.csv` (decisions + `chosen_behavior`)
- `pd_vectors_dir`: PD steering vectors `layer_*.npy`
- `split_manifest`: split indices (`train_indices`)

## What we tried (progress)

### 1) “Traditional” per-layer scalar prediction (mostly fails)

We defined:
`pref_cosine = cosine(delta_emotion, delta_pd_defect) - cosine(delta_emotion, delta_pd_cooperate)`

Then evaluated per-layer ROC-AUC to predict `chosen_behavior` (defect vs cooperate). Result:
- ROC-AUC is mostly close to chance across layers.
- Selecting a single “best layer” can give a good number, but that is vulnerable to multiple-comparisons unless layer selection is done on train and evaluated on test.

Artifacts (already generated for `20260103_164613`):
- `auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_164613/roc_auc_by_layer.csv`
- `auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_164613/roc_auc_by_layer_test_only.csv`

### 2) Mechanistic demonstration via fixed PD steering axis (works better)

Instead of predicting choices directly from a noisy scalar, we demonstrate a *mechanistic directionality*:

1) Use saved PD steering vector per layer as a fixed “defect direction”:
`v_PD(layer) := config.json["pd_vectors_dir"]/layer_{k}.npy` (unit norm in our runs)

2) Define a projection score (per item, per layer):
`score(item, layer) = dot(delta_emotion(item, layer), v_PD(layer))`

3) Summarize per item by averaging scores over that run’s controlled layers (excluding the first controlled layer if it is all-zero in deltas).

4) Report:
- `defect_rate` from `detailed_results.csv`
- `mean_projection_score` with bootstrap 95% CI (resample items)

This yields a simple, interpretable story:
- Emotion pushes the representation **toward** or **away from** the PD defect axis (sign / magnitude),
- and the direction relates to the observed behavioral shift.

## Definitions used throughout

### “test_only” split

We used the *steering-vector training split* for holdout evaluation:
- Read `train_indices` from `config.json["split_manifest"]`
- Define `test_only` as `item_id ∈ {0..499} \\ train_indices`

### Neutral baseline defect rate

From `detailed_results.csv` we compute:
- `defect_rate(neutral)` using rows where `emotion == "neutral"` and `intensity == 0.0`
- We report deltas as `Δ(defect) = defect_rate(emotion,intensity) - defect_rate(neutral,0.0)`

### Bootstrap CI for mean projection

We bootstrap over items (not over rows):
- Sample `n_items` with replacement
- Compute the mean projection score for each bootstrap sample
- CI = 2.5% and 97.5% quantiles

## Key results (high-level)

### Behavioral hypothesis
Across all analyzed runs:
- `anger` increases defect vs neutral
- `happiness` decreases defect vs neutral

### Mechanistic hypothesis (projection ↔ defect rate)
Strength depends on model size:

- **3B (`20260103_164613`)**: projection-based mechanism matches the hypothesis well:
  - `anger` has **positive** projection onto PD defect axis
  - `happiness` has **negative** projection
  - correlation(defect_rate, projection) is **positive** and significant on `test_only` across emotions/intensities

- **1.5B (`20260103_164620`, `20260103_175442`)**: behavior matches, but projection ↔ defect_rate correlation is weak.

- **0.5B (`20260103_191226`)**: behavior still matches for anger/happiness, but:
  - correlation(defect_rate, projection) becomes **negative** (mechanistic explanation fails at 0.5B under this metric/axis).

## Where the final tables live

Each run has an output file:
- `<run>/all_emotions_projection_score_with_ci.csv`

Examples:
- `auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_164613/all_emotions_projection_score_with_ci.csv`
- `auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_164620/all_emotions_projection_score_with_ci.csv`
- `auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_175442/all_emotions_projection_score_with_ci.csv`
- `auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_191226/all_emotions_projection_score_with_ci.csv`

## Reproduce (no network, uses saved artifacts)

### Environment

Activate the repo’s conda environment:
```bash
conda activate llm_fresh
```

### A) Recompute projection tables + bootstrap CI + correlations (all 4 runs)

This regenerates `<run>/all_emotions_projection_score_with_ci.csv` and prints correlations for each run.

```bash
python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RUNS = [
    Path("auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_164613"),
    Path("auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_164620"),
    Path("auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_175442"),
    Path("auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_191226"),
]

EMOTIONS = ["anger", "sadness", "happiness", "surprise", "fear", "disgust"]
INTENSITIES = [1.0, 1.5]
N_BOOT = 5000


def bootstrap_ci_mean(x: np.ndarray, n_boot: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def neutral_defect_rate(detailed: pd.DataFrame, test_ids: set[int] | None) -> float:
    sub = detailed[(detailed["emotion"] == "neutral") & (detailed["intensity"] == 0.0) & detailed["item_id"].between(0, 499)]
    if test_ids is not None:
        sub = sub[sub["item_id"].isin(test_ids)]
    return float((sub["chosen_behavior"] == "defect").mean())


for run_root in RUNS:
    cfg = json.loads((run_root / "config.json").read_text())

    model_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    if len(model_dirs) != 1:
        raise SystemExit(f"{run_root}: expected 1 model dir, got {model_dirs}")
    model_dir = model_dirs[0]

    axis_dir = Path(cfg["pd_vectors_dir"])
    axis = {int(p.stem.split("_")[1]): np.load(p).astype(np.float32) for p in axis_dir.glob("layer_*.npy")}
    if not axis:
        raise SystemExit(f"No layer_*.npy in {axis_dir}")

    man = json.loads(Path(cfg["split_manifest"]).read_text())
    train = set(man["train_indices"])
    test_ids = set(i for i in range(500) if i not in train)

    detailed = pd.read_csv(Path(cfg["result_dir"]) / "detailed_results.csv")

    # pick controlled layers from metadata when possible
    example_meta_path = model_dir / "anger" / "seed_20" / "metadata.json"
    if example_meta_path.exists():
        meta = json.loads(example_meta_path.read_text())
        controlled = [l for l in meta.get("controlled_layers", []) if l in axis]
    else:
        controlled = sorted(axis)

    # often the earliest controlled layer has all-zero deltas; drop it if so
    if controlled:
        controlled = [l for l in controlled if l != min(controlled)]

    rows = []
    for subset_name, ids in [("all", np.arange(500, dtype=int)), ("test_only", np.array(sorted(test_ids), dtype=int))]:
        for emotion in EMOTIONS:
            emo_dir = model_dir / emotion / "seed_20"
            de_all = np.load(emo_dir / "delta_emotion.npy", mmap_mode="r")

            for i_idx, intensity in enumerate(INTENSITIES):
                lab = (
                    detailed[
                        (detailed["emotion"] == emotion)
                        & (detailed["intensity"] == intensity)
                        & (detailed["item_id"].isin(ids))
                    ]
                    .sort_values("item_id")
                )
                if len(lab) != len(ids):
                    raise SystemExit(f"label mismatch: {run_root.name} {subset_name} {emotion} {intensity}: {len(lab)} vs {len(ids)}")

                y_defect = (lab["chosen_behavior"].to_numpy() == "defect").astype(np.float32)
                defect_rate = float(y_defect.mean())

                de = de_all[i_idx, ids, :, :]  # (items, layers, dim)
                proj_layers = [(de[:, l, :].astype(np.float32) @ axis[l]) for l in controlled]
                proj = np.stack(proj_layers, axis=1)
                item_score = np.nanmean(proj, axis=1)

                mean_s, lo, hi = bootstrap_ci_mean(item_score, N_BOOT)
                rows.append(
                    {
                        "subset": subset_name,
                        "emotion": emotion,
                        "intensity": intensity,
                        "n_items": int(len(ids)),
                        "defect_rate": defect_rate,
                        "layers_used": ",".join(map(str, controlled)),
                        "mean_projection_score": mean_s,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )

    out = pd.DataFrame(rows)
    out_path = run_root / "all_emotions_projection_score_with_ci.csv"
    out.to_csv(out_path, index=False)

    # correlation across (emotion,intensity) points (12 points)
    for subset_name in ["all", "test_only"]:
        s = out[out["subset"] == subset_name].copy()
        pr = pearsonr(s["defect_rate"].to_numpy(float), s["mean_projection_score"].to_numpy(float))
        sr = spearmanr(s["defect_rate"].to_numpy(float), s["mean_projection_score"].to_numpy(float))
        print(f"{run_root.name} {model_dir.name} {subset_name}: pearson r={pr[0]:.3f} p={pr[1]:.3g}; spearman rho={sr.correlation:.3f} p={sr.pvalue:.3g}")

    # print neutral baseline for context
    print(f"{run_root.name} neutral(all) defect={neutral_defect_rate(detailed, None):.4f} neutral(test_only) defect={neutral_defect_rate(detailed, test_ids):.4f}")
    print(f"wrote {out_path}")
    print()
PY
```

### B) (Optional) Recompute per-layer ROC-AUC from `pref_cosine` (example: `20260103_164613`)

This reproduces the earlier “traditional” attempt:
```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

run_root = Path("auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260103_164613")
model_dir = run_root / "Qwen2.5-3B-Instruct"
result_dir = Path("results/new_game_theory_decision/shuffle_choices/crowd-enVent_textlike_500/Qwen2.5-3B-Instruct_game_theory_decision_Prisoners_Dilemma_20251227_223627")

labels = pd.read_csv(result_dir / "detailed_results.csv")
labels = labels[(labels["emotion"] != "neutral") & labels["intensity"].isin([1.0, 1.5])]

emotions = sorted(p.name for p in model_dir.iterdir() if p.is_dir())
intensities = [1.0, 1.5]

rows = []
for emotion in emotions:
    pref = np.load(model_dir / emotion / "seed_20" / "pref_cosines.npy")  # (2,500,36)
    for i_idx, intensity in enumerate(intensities):
        lab = labels[(labels.emotion == emotion) & (labels.intensity == intensity)].sort_values("item_id")
        y = (lab["chosen_behavior"].to_numpy() == "defect").astype(int)
        for layer in range(pref.shape[2]):
            s = pref[i_idx, :, layer].astype(float)
            m = np.isfinite(s)
            auc = float("nan")
            if len(set(y[m])) == 2:
                auc = float(roc_auc_score(y[m], s[m]))
            rows.append({"emotion": emotion, "intensity": intensity, "layer": layer, "auc": auc, "n": int(m.sum())})

out = pd.DataFrame(rows)
out_path = run_root / "roc_auc_by_layer_recomputed.csv"
out.to_csv(out_path, index=False)
print("wrote", out_path)
PY
```

## Notes / cautions

- “Cherry-picking layers” can inflate AUCs; if you want a robust claim, select layers on the **train split** and evaluate on **test_only**.
- The PD-axis projection mechanism appears to depend on model size:
  - clear at 3B,
  - weak at 1.5B,
  - and breaks (negative correlation) at 0.5B for the runs examined here.

