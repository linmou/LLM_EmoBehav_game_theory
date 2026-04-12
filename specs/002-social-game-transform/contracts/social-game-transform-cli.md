# Contract: Social Game Transform CLI

## Purpose

Define the user-facing command contract for transforming curated social-game cases into game-loadable scenario datasets.

## Command

```text
python -m data_creation.transform_social_game_cases [options]
```

## Required Arguments

| Argument | Type | Description |
|--------|-------------| 
| `--social-game` | string | Social game key. V1 accepts only `beauty_contest`. |
| `--input-path` | path | Curated source case file to transform. |
| `--output-dir` | path | Directory for success, failure, skip, and metadata artifacts. |
| `--few-shot-path` | path | Few-shot asset for the selected social game. |

## Optional Arguments

| Argument | Type | Description |
|--------|-------------|
| `--rubric-path` | path | Shared rubric file path. Defaults to repository `transform_rubrics.md`. |
| `--model` | string | Chat model identifier. Defaults to the planned DeepSeek model for the run. |
| `--num-workers` | integer | Parallel worker count for API calls. |
| `--limit` | integer | Process only the first N source rows. |
| `--rerun` | flag | Ignore resume state and rebuild artifacts from scratch. |
| `--max-retries` | integer | Row-level retry count for transient prompt failures. |

## Exit Behavior

| Exit Code | Meaning |
|--------|-------------|
| `0` | Run completed with full accounting. Some rows may still be in failure/skipped artifacts. |
| non-zero | Fatal setup or unrecoverable run failure prevented trustworthy completion. |

## Standard Output

Human-readable progress lines and a final summary:

```text
social_game=beauty_contest total=2000 success=1800 failed=200 skipped=0
success_output=/abs/path/.../beauty_contest.success.json
failure_output=/abs/path/.../beauty_contest.failures.jsonl
metadata_output=/abs/path/.../run_metadata.json
```

## Standard Error

Used for fatal configuration errors, malformed input setup, missing prompt assets, or missing identity fields.

## Artifact Contract

The command writes:
- one success dataset containing only loadable transformed cases
- one failure artifact set for invalid or unsuccessful rows
- one metadata artifact for run bookkeeping and reproducibility

The command must never mix failure entries into the main success dataset.

## Validation Rules

- `--social-game beauty_contest` requires an explicit target mapping to the Beauty Contest game contract.
- Every source row must supply both `id` and `source.game_id`, or the row becomes invalid and is written to failure artifacts.
- Every successful transformed row must instantiate through the target scenario class constructor, `scenario_class(**data)`, which is `BeautyContestScenario` in V1.

## Compatibility Notes

- The CLI is internal to this repository, but its artifact structure is treated as stable for downstream experiments once implemented.
- Support for future social games requires explicit code mapping and prompt assets rather than automatic discovery.
