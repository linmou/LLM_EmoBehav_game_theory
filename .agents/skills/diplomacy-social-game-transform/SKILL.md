---
name: diplomacy-social-game-transform
description: Build, review, and scale diplomacy social-game transformation corpora that convert curated raw cases into loadable game-class scenarios. Use when adding a new social game, extending an existing transformation workflow, designing or reviewing few-shot examples, running `data_creation/transform_social_game_cases.py`, auditing generated descriptions or behavior choices, or checking lexical shortcuts in option wording with normalized prefix and first-divergence accuracy.
---

# Diplomacy Social Game Transform

## Overview

Use this skill to turn curated diplomacy social-game cases into validated corpus rows for a target `game_name`, with special attention to few-shot design, variant fidelity, and lexical-shortcut control in `behavior_choices`.

Keep the workflow brief in the skill body and make each owned file self-explainable; if one step depends on a rubric, reference, or script, point to that file instead of re-explaining its contents here.

## Workflow

1. Define the target contract first by checking the target scenario class in `games/<game_module>.py` and its registration in `games/game_configs.py`, because every transformed row must instantiate through `scenario_class` without silent fallback.

2. Use curated source rows from `input_path=/Users/admin/Documents/GitHub.nosynchr/diplomacy_cicero/social_game_outputs/<game_name>/curated_cases/<game_name>_cases.jsonl`, because the corpus must remain traceable to real raw cases.

3. Build or update `few_shot_path=<repo>/data_creation/transform_to_natural_lannguage_samples/diplomacy/<game_name>_few_shot_examples.json` as a JSON list of `{input, output}` pairs where `input` is the full raw curated case and `output` matches the target scenario shape, because this file is the strongest control surface for style, structure, variant preservation, and lexical balance.

4. Start from `assets/game_name_few_shot_examples.template.json` when adding a new game, then replace every placeholder with real raw-case content and target-schema fields instead of shrinking `input` into a summary, because abbreviated inputs weaken traceability and review quality.

5. Keep shared transformation rules in `data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md`, but prefer pushing game-specific decisions into the few-shot examples or the game validator instead of overloading the shared rubric.

6. Run the canonical transform pipeline through `python -m data_creation.transform_social_game_cases --social-game <game_name> --input-path <input_path> --few-shot-path <few_shot_path> --rubric-path <rubric_path> --output-dir <output_dir> --limit <N>`, adding `--num-workers`, `--num-candidates`, or `--temperature` only when needed, because that script owns prompt construction, model calls, runtime field injection, provenance, and contract validation.

7. Start with a small run, inspect `<output_dir>/<game_name>.success.json`, `<output_dir>/<game_name>.failures.jsonl`, `<output_dir>/<game_name>.skipped.jsonl`, and `<output_dir>/run_metadata.json`, and fix the prompt assets before scaling, because failures are usually more informative than successful rows at this stage.

8. Before any example-by-example review of a few-shot file, compute and show description diversity metrics `distinct_1`, `distinct_2`, and `distinct_3`, then compute and show both ordinary prefix shortcut accuracy (`prefix1_acc`, `prefix2_acc`) and first-divergence shortcut accuracy (`divergence1_acc`, `divergence2_acc`) after normalizing number wording to a shared placeholder such as `[x]` and collapsing participant-line phrases such as `your line` or `Italy's line` into a shared placeholder such as `[player's] line`, because the human should see corpus-level signals before spending time on row-level edits.

9. If any of `prefix1_acc`, `prefix2_acc`, `divergence1_acc`, or `divergence2_acc` is above `0.85`, stop the workflow at the few-shot stage; do not hand the file to the human, do not launch a transform run, and do not scale anything yet. Refine the few-shot examples first, preserve the semantic meaning of every option while reducing lexical leakage, and recompute the metrics, because high shortcut accuracy means the wording is still leaking the label.

10. Only after all four shortcut metrics are below `0.85` should you stop and ask the human to review few-shot examples one by one, because human review should happen on a file that already cleared the coarse shortcut gate.

11. For transformed corpora beyond the few-shot file, default to metric-level review only: inspect diversity metrics, shortcut metrics, artifact counts, and validation/failure rates, and do not switch into row-by-row human review unless the human explicitly asks for it.

12. Review descriptions as a corpus rather than row by row by looking for repeated sentence skeletons, repeated warning endings, repeated phrase bundles, and collapse of multi-turn or over-two-agent variants into bilateral single-turn prose.

13. Review `behavior_choices` as a classification surface and deliberately reduce lexical shortcuts by avoiding stable mappings like `advance/join/commit -> escalate` and `stay/keep/remain -> withdraw`; mix action families across examples and include anti-shortcut pairs where the first word is misleading unless the full phrase is read, and do not assume a shared opening fixes leakage unless the first divergence words are also uninformative.

14. Use `scripts/evaluate_prefix_shortcuts.py` in this skill when you want a quick lexical-shortcut check, but also compute first-divergence metrics after stripping the longest common prefix within each local choice set; once anti-shortcut common prefixes are introduced, treat `divergence1_acc` and `divergence2_acc` as the more meaningful signal and keep ordinary prefix metrics as a coarse baseline.

15. Stop and discuss with the human before changing any script code, including `data_creation/transform_social_game_cases.py` and any helper under `scripts/`, because script changes expand scope beyond prompt-asset curation and should not be made silently in this workflow.

16. Scale only after the small run is clean, the descriptions still reflect the source variant structure, the `behavior_choices` do not collapse into cheap lexical labels, and every successful row remains loadable through `scenario_class`.

## Lexical Shortcut Rule

Use both ordinary prefix metrics and divergence metrics.

- `prefix1_acc` and `prefix2_acc` measure whether labels can be guessed from the literal first one or two tokens.
- `divergence1_acc` and `divergence2_acc` measure whether labels can be guessed from the first one or two tokens that remain after removing the shared leading prefix inside each local choice set.
- Before scoring, replace literal numbers and common number words with a shared placeholder such as `[x]` so amount mentions do not act as cheap lexical labels.
- Before scoring, also collapse participant-specific line phrases such as `your line`, `Italy's line`, or another participant name from the row's `participants` field into one shared placeholder such as `[player's] line`, so role wording does not become a cheap lexical label.
- Low accuracy is good: the label cannot be guessed from the surface.
- High accuracy is bad: the option wording is leaking the label.
- Treat `0.85` as a hard retry threshold for both ordinary prefix metrics and divergence metrics; if any one of the four metrics exceeds it, stop at the few-shot stage, refine the few-shot wording, keep the option semantics intact, and measure again before handing the file to a human reviewer or running the transform pipeline.
- Once common prefixes are intentionally aligned across options, prioritize divergence metrics over ordinary prefix metrics.
- For a small few-shot file, the metric is a warning signal rather than a hard benchmark, so inspect the top predictive divergence prefixes together with the score.

Good anti-shortcut examples:

- `Keep pushing forces into Sweden this season.` / `Advance no further into Sweden this season.`
- `Stay committed to the advance into Marseilles this season.` / `Commit no forces to Marseilles this season.`
- `Remain in the contest for Gascony this season.` / `Enter no claim on Gascony this season.`
- `Keep adding pressure on Piedmont this season.` / `Commit only to positions outside Piedmont this season.`

## References

- Read [references/workflow.md](references/workflow.md) for the concise repo file map and command template.
- Run [scripts/evaluate_prefix_shortcuts.py](scripts/evaluate_prefix_shortcuts.py) for prefix-based lexical-shortcut scoring on few-shot files or transformed corpora.
- Copy [assets/game_name_few_shot_examples.template.json](assets/game_name_few_shot_examples.template.json) when starting a new `<game_name>_few_shot_examples.json` file.
