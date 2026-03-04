# Statistical Methods for Heatmap and HumanEval Significance

Intent: document the exact statistical procedures implemented in `result_analysis/` for significance reporting in behavior heatmaps and HumanEval emotion comparisons.

## Scope

- Heatmap significance method: `result_analysis/game_theory_impact_heatmaps.py`
- Human-alignment significance in this repo is implemented as HumanEval significance: `result_analysis/humaneval_significance.py`

Note: there is no separate `human_alignment` module name under `result_analysis`; the implemented analysis target is HumanEval (`humaneval`).

## 1) Heatmap significance (game-theory behavior direction)

### What the heatmap cell value represents

For each `(model, emotion)` cell, the displayed delta is:

- `Delta = P(target_behavior | emotion, chosen_intensity) - P(target_behavior | neutral_baseline)`

where:

- `target_behavior` is chosen by rule:
- Binary games: prefer `defect` or `escalation` if present; otherwise the second non-unknown label.
- Multi-behavior games: prefer `offer_none` or `reject` if present; otherwise the last non-unknown label.
- `chosen_intensity` is the emotion intensity with maximal `|Delta|` (tie-break: lower intensity).
- `neutral_baseline` is the mean target-behavior ratio across available neutral intensities.

### Significance test per cell

Per `(model, emotion)` at the selected intensity, we build paired binary outcomes on matched `(item_id, repeat_id)`:

- `n_i = 1` if neutral choice is target behavior; else `0`
- `e_i = 1` if emotion choice is target behavior; else `0`

Only discordant pairs enter McNemar:

- `n01`: pairs with `(n_i, e_i) = (0, 1)`
- `n10`: pairs with `(n_i, e_i) = (1, 0)`

The code uses exact two-sided McNemar p-value from a binomial tail:

- `n = n01 + n10`
- `p = 2 * sum_{i=0..min(n01,n10)} C(n, i) / 2^n`, clipped at `1.0`

### Multiple-testing correction

Across all tested `(model, emotion)` cells in that heatmap computation, p-values are corrected with Benjamini-Hochberg FDR:

- Rank p-values ascending.
- Compute `q_i = min(1, p_i * m / rank_i)`.
- Enforce monotone non-increasing adjusted q from largest p to smallest p.

### Annotation in plots

Each cell text is:

- signed delta to 2 decimals, plus stars from FDR-adjusted q:
- `***` for `q < 0.001`
- `**` for `q < 0.01`
- `*` for `q < 0.05`
- no star otherwise

## 2) HumanEval significance (used as human-alignment proxy here)

### Pairing strategy

Within each run directory (`results/humaneval/<model_run>/detailed_results.csv`), for each emotion:

- Match items by `item_id` intersection with neutral.
- Build paired differences:
- `d_i = score_emotion_i - score_neutral_i`

### Test statistic

The implementation uses a paired t-test statistic:

- `d_bar = mean(d_i)`
- `sd_d = sample std of d_i`
- `t = d_bar / (sd_d / sqrt(n))`, with `df = n - 1`

Special case:

- if `sd_d == 0` and `d_bar != 0`, `t` is treated as signed infinity.
- if both are zero, `t = 0`.

### Significance decision rule

- Two-sided `alpha = 0.05`.
- Compare `|t|` against `t_critical(df)` from an internal lookup table with linear interpolation for missing df.
- Mark significant when `|t| >= t_critical`.

### Reported quantities

For each emotion vs neutral:

- `Delta pass@1` (percentage-point difference),
- `t` statistic,
- significance flag (star),
- emotion mean score (`neutral_mean + mean_delta`).

## Practical interpretation notes

- Heatmap significance is paired at the item level and controls FDR across cells, so it is robust to multiple comparisons in that panel.
- HumanEval significance is paired across problems but does not apply additional multiple-testing correction across emotions/models in the current script.
- Both analyses are intentionally simple and direct, matching the KISS style in this repository.
