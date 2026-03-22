# Result Analysis Directory

Last updated: 2026-03-21

This directory contains all post-experiment analysis scripts and results for the LLM Emotional Behavior Game Theory experiments.

## Analysis Scripts

### Core Analysis Scripts
- `analyze_switches_detailed.py` - Analyzes switching patterns between activation_only and context_and_activation conditions
- `analyze_choice_patterns.py` - Comprehensive analysis of choice patterns across all conditions
- `analyze_choice_differences.py` - Finds cases where choices differ between conditions
- `generate_game_theory_impact_report.py` - Builds option/behavior impact tables vs neutral (plus intensity tables, filtering, and heatmaps)
- `postprocess_prob_argmax_from_existing_csv.py` - Postprocess existing argmax-match CSV with behavior labels + predicted argmax distributions
- `trust_game_trustor_expected_score.py` - Report-driven Trust Game (Trustor) item-level decision shift vs neutral (trust_none=0, trust_low=1, trust_high=2)
- `trust_game_trustee_expected_score.py` - Report-driven Trust Game (Trustee) item-level decision shift vs neutral (return_none=0, return_medium=1, return_high=2)
- `trust_game_expected_score.py` - Shared Trust Game expected-score analysis (runs both roles by default; use `--role` to limit)
- `ultimatum_game_expected_score.py` - Ultimatum Game expected-score analysis (runs proposer+responder by default; use `--role` to limit)

### Debug/Utility Scripts
- `debug_data_structure.py` - Utility to inspect data structure and verify scenario matching

## Analysis Results

### Switching Pattern Results
- `detailed_switches_with_prompts.csv` - Complete switching data including input prompts and model outputs (370KB)
  - Contains 119 cases where choices switched between activation_only and context_and_activation
  - Includes full prompts and generated texts for detailed analysis
  
- `detailed_switches_ao_ca.csv` - Summary of switching patterns without full prompts (26KB)

## Key Findings

From the latest experiment (2025-07-03):
- **Total scenarios tested**: 1000 (same scenarios across all 4 conditions)
- **Switching cases**: 119 (11.9% of scenarios showed different choices)
  - 67 switches from cooperation to defection (1→2)
  - 52 switches from defection to cooperation (2→1)

### Defection Rates by Condition:
- Baseline: 5.9%
- Context only: 5.3%
- Activation only (anger): 6.7%
- Context + Activation (anger): 8.1%

## Usage

To run analysis on new experiment results:

```bash
# Update the file path in the script to point to new results
python analyze_switches_detailed.py

# For comprehensive pattern analysis
python analyze_choice_patterns.py

# Trust Game expected-score deltas from a series report (Trustor + Trustee)
python -m result_analysis.trust_game_expected_score \
  --report results/.../memory_experiment_series_..._memory_experiment_report.json \
  --out_dir results/.../shuffle_decision_only

This writes both `trustor_*` and `trustee_*` outputs when the report contains both benchmarks.

# Inspecting Individual Decisions from Aggregated Ratios

The experiment runner writes a `raw_results.json` file alongside summary CSVs (including `summary_choice_ratio.csv` and, when available, `summary_behavior_ratio.csv`). A simple pattern to trace an aggregate ratio back to an example decision is:

```python
import json
from pathlib import Path

results_dir = Path("path/to/experiment/results")
raw = json.loads((results_dir / "raw_results.json").read_text(encoding="utf-8"))

# Pick any decision matching an aggregate row, e.g. emotion="anger", intensity=0.1
sample = next(
    r
    for r in raw
    if r.get("emotion") == "anger" and float(r.get("intensity", 0.0)) == 0.1
)

meta = sample.get("metadata") or {}
item_md = meta.get("item_metadata") or {}
options = item_md.get("options") or []

chosen_id = int(sample["score"])
chosen_behavior = next(
    opt["behavior"]
    for opt in options
    if int(opt["id"]) == chosen_id
)

print("Chosen option id:", chosen_id)
print("Chosen behavior:", chosen_behavior)
print("Options:", options)
```

This keeps analysis in pure Python (no new dependencies) and matches the dataset’s metadata format used for behavior-level choice ratios.
```

## Game-Theory Impact Report (vs neutral)

This aggregates per-game choice/behavior ratios and reports emotion deltas vs `neutral`, collapsing over intensity (with intensity-aware tables too).

```bash
# Shuffle-choice decision benchmark (choice + behavior ratios)
python -m result_analysis.generate_game_theory_impact_report \
  --root results/new_game_theory_decision/shuffle_choices

# Older game-theory benchmark (usually choice ratios only)
python -m result_analysis.generate_game_theory_impact_report \
  --root results/new_game_theory
```

This writes into the `--root` folder:
- `option_impacted_by_emo_vs_neutral_latest.csv`
- `behavior_impacted_emo_vs_neutral_latest.csv` (only if `summary_behavior_ratio.csv` exists)
- `option_intensity_impacted_by_emo_vs_neutral_latest.csv`
- `behavior_intensity_impacted_emo_vs_neutral_latest.csv` (only if `summary_behavior_ratio.csv` exists)
- `game_theory_impact_report.md` (includes which runs were used + any skipped runs missing `neutral`)

Optional:
- Use `--out_dir` when `--root` is not writable (writes outputs to `--out_dir` but still scans `--root`).
- Use `--unknown_threshold 0.10` to drop `(emotion,intensity)` slices with high unknown ratio (behavior: `behavior=="unknown"`, choice: `option_id==-1`).
- Use `--write_heatmaps` to write one behavior-change heatmap PDF per game setting under `out_dir/heatmaps/`.
  - Heatmap cells are peak-`|Δ|` across intensities per `(model, emotion)` for a target “direction” behavior (binary: `defect/escalation`; otherwise: `offer_none/reject`).
  - Default heatmap normalization is symmetric-log; tune with `--heatmap_symlog_linthresh` or override via `--heatmap_norm linear`.

## Original Experiment Results Location

The raw experiment results are stored in:
```
results/Choice_Selection/choice_selection_choice_selection_context_activation_test_prisoners_dilemma_Qwen2.5-3B-Instruct_20250703_133259/
```
