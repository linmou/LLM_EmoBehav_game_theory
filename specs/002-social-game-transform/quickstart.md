# Quickstart: Social Game Case Transformation Pipeline

## Purpose

Run the first-release transformation workflow for both supported social games and verify that each produces a success-only dataset loadable by the corresponding game scenario class.

## Prerequisites

1. Ensure the repository Python environment with `openai`, `python-dotenv`, `pydantic`, and the in-repo game modules is active.
2. Ensure `.env` contains the DeepSeek-compatible credentials required for the run, especially `DPSK_API`.
3. Ensure the shared rubric asset exists:

```text
/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/transform_rubrics.md
```

4. Ensure the curated source file exists for the selected social game.
5. Ensure the few-shot asset path exists. In this release, `escalation_game` may temporarily reuse the same Beauty Contest few-shot asset.

## Example Run: `beauty_contest`

```bash
python -m data_creation.transform_social_game_cases \
  --social-game beauty_contest \
  --input-path /home/jjl7137/diplomacy_cicero/social_game_outputs/beauty_contest/curated_cases/beauty_contest_cases.jsonl \
  --few-shot-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/beauty_contest_few_shot_examples.json \
  --rubric-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/transform_rubrics.md \
  --output-dir /tmp/beauty_contest_transform_run
```

Expected output files:

```text
beauty_contest.success.json
beauty_contest.failures.jsonl
beauty_contest.skipped.jsonl
run_metadata.json
```

Verification:

1. Confirm the success dataset contains only valid transformed rows.
2. Confirm failures and skips are written only to their dedicated artifacts.
3. Confirm `run_metadata.json` counters reconcile to the total processed rows.
4. Load a sample of success rows through `BeautyContestScenario(**data)`.

## Example Run: `escalation_game`

```bash
python -m data_creation.transform_social_game_cases \
  --social-game escalation_game \
  --input-path /home/jjl7137/diplomacy_cicero/social_game_outputs/escalation_game/curated_cases/escalation_game_cases.jsonl \
  --few-shot-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/beauty_contest_few_shot_examples.json \
  --rubric-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/transform_rubrics.md \
  --output-dir /tmp/escalation_game_transform_run
```

Expected output files:

```text
escalation_game.success.json
escalation_game.failures.jsonl
escalation_game.skipped.jsonl
run_metadata.json
```

Verification:

1. Confirm the success dataset contains only rows loadable through `EscalationGameScenario(**data)`.
2. Confirm canonical `game_name` and payoff data are injected deterministically.
3. For rows with explicit `previous_actions`, confirm they validate and remain intact.
4. For rows with both `previous_actions` and `previous_actions_length`, confirm mismatches fail and land in failure artifacts rather than being silently repaired.
5. During the temporary first-release reuse of the Beauty Contest few-shot asset, expect some live rows to fail the Escalation Game contract and appear only in `escalation_game.failures.jsonl`; that is evidence of contract enforcement, not silent degradation.

## Resume Check

1. Re-run either command without `--rerun`.
2. Confirm previously completed identities are not duplicated.
3. Confirm resumed rows produce entries in the skipped artifact.

## Test Focus

- Explicit mapping supports both `beauty_contest` and `escalation_game`.
- Unsupported games still fail loudly.
- Prompt assembly uses the shared rubric plus the selected few-shot asset.
- Success datasets remain success-only.
- Resume logic keys on `id + source.game_id`.
- Beauty Contest rows instantiate through `BeautyContestScenario(**data)`.
- Escalation rows instantiate through `EscalationGameScenario(**data)`.
- `Escalation_Game` history obeys the optional-explicit-history plus fallback-length contract.
