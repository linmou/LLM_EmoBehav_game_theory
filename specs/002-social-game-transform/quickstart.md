# Quickstart: Social Game Case Transformation Pipeline

## Purpose

Run the transformation workflow for the supported social games and verify that:
- successful outputs still load through the real scenario class
- few-shot selection is same-game only
- each source row keeps at least one same-variant few-shot example
- each multi-variant source row gets a per-row few-shot pack with exactly 2 cross-variant examples
- diversity artifacts are written for audit

## Prerequisites

1. Ensure the repository Python environment with `openai`, `python-dotenv`, and the in-repo game modules is active.
2. Ensure `.env` contains the DeepSeek-compatible credentials required for the run, especially `DPSK_API`.
3. Ensure the shared rubric asset exists:

```text
/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md
```

4. Ensure the curated source file exists for the selected social game.
5. Ensure the selected few-shot asset exists for the same social game as the run:

```text
/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_to_natural_lannguage_samples/diplomacy/beauty_contest_few_shot_examples.json
/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_to_natural_lannguage_samples/diplomacy/escalation_game_few_shot_examples.json
```

## Example Run: `beauty_contest`

```bash
python -m data_creation.transform_social_game_cases \
  --social-game beauty_contest \
  --input-path /Users/admin/Documents/GitHub.nosynchr/diplomacy_cicero/social_game_outputs/beauty_contest/curated_cases/beauty_contest_cases.jsonl \
  --few-shot-path /Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_to_natural_lannguage_samples/diplomacy/beauty_contest_few_shot_examples.json \
  --rubric-path /Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md \
  --output-dir /tmp/beauty_contest_transform_run \
  --num-candidates 4 \
  --temperature 0.7
```

Expected output files:

```text
beauty_contest.success.json
beauty_contest.failures.jsonl
beauty_contest.skipped.jsonl
beauty_contest.candidates.jsonl
diversity_report.json
run_metadata.json
```

Verification:

1. Confirm the success dataset contains only valid transformed rows.
2. Confirm failures and skips are written only to their dedicated artifacts.
3. Confirm `run_metadata.json` counters reconcile to the total processed rows.
4. Confirm `diversity_report.json` contains classic n-gram metrics for selected outputs.
5. Load a sample of success rows through `BeautyContestScenario(**data)`.

## Example Run: `escalation_game`

```bash
python -m data_creation.transform_social_game_cases \
  --social-game escalation_game \
  --input-path /Users/admin/Documents/GitHub.nosynchr/diplomacy_cicero/social_game_outputs/escalation_game/curated_cases/escalation_game_cases.jsonl \
  --few-shot-path /Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_to_natural_lannguage_samples/diplomacy/escalation_game_few_shot_examples.json \
  --rubric-path /Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md \
  --output-dir /tmp/escalation_game_transform_run \
  --num-candidates 4 \
  --temperature 0.7
```

Expected output files:

```text
escalation_game.success.json
escalation_game.failures.jsonl
escalation_game.skipped.jsonl
escalation_game.candidates.jsonl
diversity_report.json
run_metadata.json
```

Verification:

1. Confirm the success dataset contains only rows loadable through `EscalationGameScenario(**data)`.
2. Confirm canonical `game_name` and payoff data are injected deterministically.
3. For rows with explicit `previous_actions`, confirm they validate and remain intact.
4. For rows with both `previous_actions` and `previous_actions_length`, confirm mismatches fail and land in failure artifacts rather than being silently repaired.
5. Confirm the run uses the `escalation_game` few-shot asset rather than a different social game's asset.

## Few-Shot Selection Audit

1. Inspect the run input and identify the variants present in that batch.
2. Confirm the few-shot selection pool only contains examples from the same social game and from those run-present variants.
3. Confirm each source row's selected pack keeps at least one same-variant example.
4. For multi-variant runs, confirm each source row's selected pack contains exactly 2 cross-variant examples and fills the remaining slots from the row's own variant.
5. Confirm lexical scoring only considers `description` and `behavior_choices`.
6. Confirm the ranking is reproducible when rerun against the same eligible pool.

## Resume Check

1. Re-run either command without `--rerun`.
2. Confirm previously completed identities are not duplicated.
3. Confirm resumed rows produce entries in the skipped artifact.

## Test Focus

- Explicit mapping supports both `beauty_contest` and `escalation_game`.
- Unsupported games still fail loudly.
- Prompt assembly uses the shared rubric plus the selected same-game few-shot asset.
- Per-row pack construction enforces the same-variant anchor rule and, for multi-variant runs, the exact same-variant-plus-2-cross-variant rule.
- Success datasets remain success-only.
- Resume logic keys on `id + source.game_id`.
- Beauty Contest rows instantiate through `BeautyContestScenario(**data)`.
- Escalation rows instantiate through `EscalationGameScenario(**data)`.
- Classic n-gram diversity metrics are emitted as audit artifacts.
