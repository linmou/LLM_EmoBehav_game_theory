# Quickstart: Social Game Case Transformation Pipeline

## Purpose

Run the first-release `beauty_contest` transformation workflow end to end and verify that it produces a success-only dataset loadable by the Beauty Contest game class.

## Prerequisites

1. Ensure the repository Python environment with `openai`, `python-dotenv`, and the game modules is active. On the current machine, the working environment is `llm`.
2. Ensure `.env` contains the DeepSeek-compatible credentials required for the run.
3. Ensure the curated source file exists:

```text
/home/jjl7137/diplomacy_cicero/social_game_outputs/beauty_contest/curated_cases/beauty_contest_cases.jsonl
```

4. Ensure the few-shot asset for `beauty_contest` exists and is mapped in the implementation.

## Example Run

```bash
python -m data_creation.transform_social_game_cases \
  --social-game beauty_contest \
  --input-path /home/jjl7137/diplomacy_cicero/social_game_outputs/beauty_contest/curated_cases/beauty_contest_cases.jsonl \
  --few-shot-path /abs/path/to/beauty_contest_few_shot_examples.json \
  --output-dir /abs/path/to/output/beauty_contest_run \
  --num-workers 8
```

## Expected Outputs

The output directory should contain:

```text
beauty_contest.success.json
beauty_contest.failures.jsonl
beauty_contest.skipped.jsonl
run_metadata.json
```

## Verification Steps

1. Confirm the main success dataset contains only successful transformed cases.
2. Confirm failed or skipped rows appear only in separate artifacts.
3. Confirm `run_metadata.json` counters reconcile to the total number of processed source rows.
4. Load a sample of success rows through direct scenario construction with the Beauty Contest game scenario class.
5. Re-run the command without `--rerun` and confirm previously completed identities are not duplicated.

## Test Focus

- Prompt assembly uses both the shared rubric and the mapped few-shot asset.
- Resume logic keys on `id + source.game_id`.
- Invalid rows are accounted for without polluting the main success dataset.
- Successful rows instantiate through `BeautyContestScenario(**data)`.
