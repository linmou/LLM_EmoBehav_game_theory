# Experiment Results Files

This folder contains outputs from EmotionExperiment. Files:

- detailed_results.csv: Item-level records including emotion, intensity, repeat_id, response, ground_truth, score.
- raw_results.json: Full JSON dump of all records with metadata (benchmark, item metadata).
- summary_results.csv: Aggregates per (emotion,intensity) across all repeats (mean, std, count, min, max).
- summary_by_repeat.csv: Aggregates per (emotion,intensity,repeat_id).
- summary_overall.csv: Across-repeat statistics per (emotion,intensity):
  - mean_of_means: Unweighted mean of per-repeat means.
  - between_run_var: Sample variance of per-repeat means (repeat-level stability).
  - pooled_var: Unbiased pooled variance across all observations (law of total variance).
- summary_choice_ratio.csv: Per-option selection ratios grouped by emotion and intensity (present when dataset supplies choice ratios).
- summary_choice_ratio_by_repeat.csv: Per-option selection ratios grouped by emotion, intensity, and repeat (requires dataset-supplied choice ratios and repeat runs).
- experiment_config.json: Resolved configuration and runtime info (includes repeat settings).

Notes:
- For meaningful repeat variance, enable stochastic decoding (do_sample=true, nonzero temperature/top_p).
- Seeds: Each repeat uses random_seed = repeat_seed_base + repeat_id when supported.
