# Diplomacy Social Game Transform Workflow

Intent: keep the repo-specific file map and command template in one short place so the skill body can stay procedural and lean.

Rule: keep each listed file self-explainable in its own scope, and do not duplicate detailed explanations across the workflow note, rubric, few-shot file, and transform script.

## File Map

- `games/<game_module>.py`: target scenario class and validation logic
- `games/game_configs.py`: `game_name` registration and `scenario_class` lookup
- `data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md`: shared transform rules for diplomacy-sourced corpora
- `data_creation/transform_to_natural_lannguage_samples/diplomacy/<game_name>_few_shot_examples.json`: game-specific `{input, output}` exemplars
- `data_creation/transform_social_game_cases.py`: canonical transform pipeline
- `results/transform_runs/<run_name>/`: run artifacts and audit trail

## Command Template

```bash
python -m data_creation.transform_social_game_cases \
  --social-game <game_name> \
  --input-path /Users/admin/Documents/GitHub.nosynchr/diplomacy_cicero/social_game_outputs/<game_name>/curated_cases/<game_name>_cases.jsonl \
  --output-dir <repo>/results/transform_runs/<run_name> \
  --limit <N>
```

Optional explicit override:

```bash
python -m data_creation.transform_social_game_cases \
  --social-game <game_name> \
  --input-path /Users/admin/Documents/GitHub.nosynchr/diplomacy_cicero/social_game_outputs/<game_name>/curated_cases/<game_name>_cases.jsonl \
  --few-shot-path <repo>/data_creation/transform_to_natural_lannguage_samples/diplomacy/<game_name>_few_shot_examples.json \
  --rubric-path <repo>/data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md \
  --output-dir <repo>/results/transform_runs/<run_name> \
  --limit <N>
```

## Review Targets

- `success.json`: loadable transformed rows only
- `failures.jsonl`: rows that failed prompt or validation
- `skipped.jsonl`: resumed rows skipped because they were already finalized
- `run_metadata.json`: counts, input/output paths, prompt asset paths, and completed identities

## Shortcut Audit

Run:

```bash
python .agents/skills/diplomacy-social-game-transform/scripts/evaluate_prefix_shortcuts.py <json_path>
```

Supported inputs:

- few-shot file shaped like `[{input, output}, ...]`
- transformed corpus shaped like `[{behavior_choices, ...}, ...]`

Interpretation:

- low first-token and first-two-token accuracy is good
- high first-token or first-two-token accuracy means the choices are leaking the label through prefixes
