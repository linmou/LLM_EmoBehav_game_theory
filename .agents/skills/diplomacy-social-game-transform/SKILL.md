---
name: diplomacy-social-game-transform
description: Build, review, and scale diplomacy social-game transformation corpora that convert curated raw cases into loadable game-class scenarios. Use when adding a new social game, extending an existing transformation workflow, designing or reviewing few-shot examples, running `data_creation/transform_social_game_cases.py`, auditing generated descriptions or behavior choices, or checking lexical shortcuts in option wording with first-token or first-two-token accuracy.
---

# Diplomacy Social Game Transform

## Overview

Use this skill to turn curated diplomacy social-game cases into validated corpus rows for a target `game_name`, with special attention to few-shot design, variant fidelity, and lexical-shortcut control in `behavior_choices`.

Keep the workflow brief in the skill body and make each owned file self-explainable; if one step depends on a rubric, reference, or script, point to that file instead of re-explaining its contents here.

## Workflow

1. Define the target contract first by checking the target scenario class in `games/<game_module>.py` and its registration in `games/game_configs.py`, because every transformed row must instantiate through `scenario_class` without silent fallback.

2. Use curated source rows from `input_path=/Users/admin/Documents/GitHub.nosynchr/diplomacy_cicero/social_game_outputs/<game_name>/curated_cases/<game_name>_cases.jsonl`, because the corpus must remain traceable to real raw cases.

3. Build or update `few_shot_path=<repo>/<game_name>_few_shot_examples.json` as a JSON list of `{input, output}` pairs where `input` is the full raw curated case and `output` matches the target scenario shape, because this file is the strongest control surface for style, structure, variant preservation, and lexical balance.

4. Keep shared transformation rules in `data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md`, but prefer pushing game-specific decisions into the few-shot examples or the game validator instead of overloading the shared rubric.

5. Run the canonical transform pipeline through `python -m data_creation.transform_social_game_cases --social-game <game_name> --input-path <input_path> --few-shot-path <few_shot_path> --rubric-path <rubric_path> --output-dir <output_dir> --limit <N>`, adding `--num-workers`, `--num-candidates`, or `--temperature` only when needed, because that script owns prompt construction, model calls, runtime field injection, provenance, and contract validation.

6. Start with a small run, inspect `<output_dir>/<game_name>.success.json`, `<output_dir>/<game_name>.failures.jsonl`, `<output_dir>/<game_name>.skipped.jsonl`, and `<output_dir>/run_metadata.json`, and fix the prompt assets before scaling, because failures are usually more informative than successful rows at this stage.

7. Review descriptions as a corpus rather than row by row by looking for repeated sentence skeletons, repeated warning endings, repeated phrase bundles, and collapse of multi-turn or over-two-agent variants into bilateral single-turn prose.

8. Review `behavior_choices` as a classification surface and deliberately reduce lexical shortcuts by avoiding stable mappings like `advance/join/commit -> escalate` and `stay/keep/remain -> withdraw`; mix action families across examples and include anti-shortcut pairs where the first word is misleading unless the full phrase is read.

9. Use `scripts/evaluate_prefix_shortcuts.py` in this skill when you want a quick lexical-shortcut check, and treat first-token and first-two-token accuracy as the default sufficient metric unless a specific project needs something more complex.

10. Scale only after the small run is clean, the descriptions still reflect the source variant structure, the `behavior_choices` do not collapse into cheap lexical labels, and every successful row remains loadable through `scenario_class`.

## Lexical Shortcut Rule

Use first-token and first-two-token accuracy as the default shortcut metric.

- Low accuracy is good: the label cannot be guessed from a prefix.
- High accuracy is bad: the option wording is leaking the label.
- For a small few-shot file, the metric is a warning signal rather than a hard benchmark, so inspect the top predictive prefixes together with the score.

Good anti-shortcut examples:

- `Keep pushing forces into Sweden this season.` / `Advance no further into Sweden this season.`
- `Stay committed to the advance into Marseilles this season.` / `Commit no forces to Marseilles this season.`
- `Remain in the contest for Gascony this season.` / `Enter no claim on Gascony this season.`
- `Keep adding pressure on Piedmont this season.` / `Commit only to positions outside Piedmont this season.`

## References

- Read [references/workflow.md](references/workflow.md) for the concise repo file map and command template.
- Run [scripts/evaluate_prefix_shortcuts.py](scripts/evaluate_prefix_shortcuts.py) for prefix-based lexical-shortcut scoring on few-shot files or transformed corpora.
