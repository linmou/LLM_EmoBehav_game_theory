# PD-Probe Transfer Test Spec (Per-Layer Linear SVM)

Date: 2026-01-04

## Why this test exists

We already observe a behavioral effect (e.g., `anger → defect`, `happiness → cooperate`). The open question is whether the *stored internal artifacts* (hidden states / delta activations) support a mechanistic story:

> “Emotion steering induces an activation delta that aligns with the model’s PD decision manifold, and that alignment predicts the model’s actual choice.”

This spec defines a leakage-resistant, per-layer probe test that uses **only** existing artifacts:

- PD steering deltas: `delta_pd`, `delta_pd_cooperate`
- Emotion steering deltas: `delta_emotion`
- Behavioral labels: `chosen_behavior` from `detailed_results.csv`
- Item split: `split_manifest`

No refactoring and no new model runs are required.

## Research hypotheses

**H1 (Behavioral):**
- `anger` increases defect rate; `happiness` increases cooperate rate.

**H2 (Mechanistic transfer):**
- A per-layer linear probe trained to distinguish **PD-defect** vs **PD-cooperate** delta activations transfers to emotion deltas:
  - the probe score on `delta_emotion(item, layer)` predicts the model’s actual `chosen_behavior` (`defect` vs `cooperate`) on the same item.

**H3 (Layer localization):**
- Predictive power (ROC-AUC) is higher in controlled layers and/or post-control layers than in early layers.

**Known/expected failure mode (small models):**
- For 0.5B runs, transfer may weaken or invert (AUC≈0.5 or <0.5), indicating misalignment between emotion deltas and the PD decision manifold.

## Inputs and where they live

Per run directory:
`auto_experiments/task_similarity/results/emotion_pd_delta_similarity/<run_id>/`

### Delta tensors (per emotion/seed)
In:
`<run_id>/<model>/<emotion>/seed_<seed>/`

Files:
- `delta_pd.npy` with shape `(n_int, n_items, n_layers, d)`
- `delta_pd_cooperate.npy` with shape `(n_int, n_items, n_layers, d)`
- `delta_emotion.npy` with shape `(n_int, n_items, n_layers, d)`
- `metadata.json` (contains `controlled_layers`, `intensities`, `split_manifest`, `raw_results_path`)

Notes:
- We do **not** assume symmetry `delta_pd_cooperate = -delta_pd`. Due to nonlinear transformer blocks and propagation into post-control layers, these can be meaningfully asymmetric.
- In practice, `delta_pd` / `delta_pd_cooperate` are the same across emotion folders within the same run/seed (since they’re driven by PD vectors, not emotion vectors). Train once per run/seed and reuse across all emotions.

### Behavioral labels
`metadata.json["raw_results_path"]` points to a directory that contains:
- `detailed_results.csv` with columns:
  - `emotion`, `intensity`, `item_id`, `chosen_behavior` in `{defect, cooperate}`

We use `chosen_behavior` as the binary label because it already accounts for option order randomization (raw JSON `decision` text does not).

### Split manifest (leakage control)
Use `metadata.json["split_manifest"]` to define:
- `train_items`: items used to train probes
- `test_only_items`: items used only for evaluation

## Experimental design

### 1) Split protocol (no leakage)

Even though probes are trained on PD deltas (not emotion deltas), they still see item-specific representations. To avoid “memorizing items”, enforce:

- Fit probes only on `train_items`
- Evaluate transfer metrics only on `test_only_items`

### 2) Per-layer probe training task (PD deltas)

For each layer `ℓ`, train a **linear SVM**:

- Positive class (defect-like PD steer effect): `X+ = delta_pd[:, train_items, ℓ, :]`
- Negative class (cooperate-like PD steer effect): `X- = delta_pd_cooperate[:, train_items, ℓ, :]`
- Stack features: `X = [X+; X-]`
- Labels: `y = [1...; 0...]`

Intensity handling (choose one; default is simplest):
- **Default**: pool all intensities as additional training samples
- If needed: train one probe per intensity (only if pooling breaks stability)

Model choice:
- linear SVM (no kernel). Keep it boring.

### 3) Feature normalization (direction-focused default)

Default normalization:
- L2-normalize each sample vector `x ← x / (||x|| + eps)`

Rationale:
- prevents the probe from winning by magnitude differences (norm cheats)
- aligns with the “directional manifold” interpretation used by cosine/projection methods

### 4) Score orientation (consistent “defectness”)

After training each layer probe, enforce a consistent sign convention:

- Compute mean PD scores on train:
  - `m+ = mean s(delta_pd)`
  - `m- = mean s(delta_pd_cooperate)`
- If `m+ <= m-`, flip `s ← -s`.

This guarantees: higher score = more defect-like, without touching emotion/behavior labels.

### 5) Transfer evaluation: emotion delta → behavior

For each `(emotion, intensity)` and each layer `ℓ`:

- Score emotion deltas on `test_only_items`:
  - `s_emotion[item] = s(delta_emotion[intensity, item, ℓ, :])`
- Labels from `detailed_results.csv`:
  - `y_behavior[item] = 1` if `chosen_behavior == defect` else `0`
- Metric:
  - ROC-AUC(`s_emotion`, `y_behavior`) per layer per (emotion,intensity)

This is the main mechanistic result: a PD-trained probe reads out defectness from emotion deltas and predicts the model’s actual choice.

### 6) Controlled-layer summaries (no cherry-picking)

Let `controlled_layers` come from `metadata.json`.

Report both:
- Full curve: `AUC(layer)` across all layers (diagnostic)
- Aggregates over controlled layers:
  - mean AUC over `controlled_layers`
  - median AUC over `controlled_layers`

Optional (propagation): aggregate over post-control layers `layer > max(controlled_layers)`.

Hard rule:
- Do not report “best layer” unless it is selected using **train split only** and then evaluated once on test-only.

### 7) Uncertainty (bootstrap CI)

For each `(emotion,intensity,layer)`:
- Bootstrap `test_only_items` with replacement, recompute AUC, take 2.5/97.5 percentiles as 95% CI.

For mean/median AUC across controlled layers:
- In each bootstrap replicate:
  - compute per-layer AUC on resampled items
  - compute mean/median over controlled layers
- Take percentiles over replicates.

### 8) Required sanity checks

These must pass before interpreting transfer AUC:

1) **PD separability sanity** (internal control)
   - On `test_only_items`, the probe should strongly separate `delta_pd` vs `delta_pd_cooperate` (AUC >> 0.5).

2) **Neutral sanity** (if neutral is present in behavior CSV)
   - Transfer AUC should be ~0.5 for neutral (or at least much weaker than anger/happiness).

3) **Intensity stability** (optional)
   - Train on PD deltas at 1.0 only; test PD separability at 1.5.
   - If unstable, switch to per-intensity probes.

## Expected outcomes / decision criteria

Evidence supporting H2:
- For anger/happiness, controlled-layer (or post-control) mean/median transfer AUC > 0.5, ideally with CI not crossing 0.5.

Evidence supporting H3:
- Controlled/post-control summaries exceed early-layer summaries.

Evidence of small-model failure:
- Transfer AUC ≈ 0.5 or < 0.5 across controlled/post-control layers (potentially inverted) despite PD-separability sanity being strong.

## Deliverables (files)

Per run/seed, produce:
- `pd_probe_transfer_auc_by_layer.csv`
  - columns: `emotion`, `intensity`, `layer`, `auc`, `ci_low`, `ci_high`, `n_test_items`
- `pd_probe_transfer_controlled_layer_summary.csv`
  - columns: `emotion`, `intensity`, `defect_rate`, `mean_auc_controlled`, `mean_ci_low`, `mean_ci_high`, `median_auc_controlled`, `median_ci_low`, `median_ci_high`

All outputs must record:
- run_id, model identifier, seed, intensities used, controlled_layers, and split_manifest path.

