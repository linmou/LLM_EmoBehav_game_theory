# Purpose

This contract defines the required behavior of `result_analysis/generate_game_theory_impact_report.py`.
It keeps the reporting path narrow, reproducible, and resistant to malformed raw outputs.

## Inputs

- Run discovery is keyed by timestamped run-directory names matching the game-theory naming pattern.
- A report root may contain both choice and behavior summary CSVs.
- `raw_results.json` is optional for significance and may be malformed after the first valid JSON value.

## Required Behaviors

- Select the latest run per `(model, task)` within the report root.
- Compute deltas against the neutral baseline.
- Preserve per-intensity outputs in the `*_intensity_*` CSVs.
- Recover the first valid top-level JSON array from `raw_results.json` if trailing junk exists.
- Parse significance decisions from:
  - JSON strings containing a `decision`
  - dict-shaped `response` objects containing a `decision`
  - Python-dict-style strings containing a `decision`
- Ignore trivial terminal punctuation when matching a decision string back to an option text.
- For significance annotations, use the emotion intensity with peak absolute delta for that option/behavior row.
- In markdown per-game tables, keep all collapsed emotions even if significance exists for only a subset of them.
- Use significance decorations only for the subset where significance could actually be computed.

## Run Discovery Rules

- Discover top-level timestamped run directories under the report root.
- Treat top-level symlinked run directories the same as normal run directories.
- Do not let nested artifact folders such as `heatmaps/` override the real run selection.

## Output Rules

- Always write the option CSV outputs when choice ratios with neutral rows exist.
- Write behavior CSV outputs only when behavior ratio inputs exist.
- Mark skipped runs lacking neutral baselines in the markdown report.
- When significance cannot be extracted, fall back to ratio-derived deltas rather than leaving cells blank.

## Non-Goals

- Silent fallback to unrelated directories or alternate configs.
- Reinterpreting malformed raw outputs beyond recovering the first valid top-level JSON array.
