# Contract: Social Game Transform CLI

## Purpose

Define the user-facing CLI contract for transforming curated social-game source rows into success-only datasets loadable by the repository’s real game scenario classes.

## Command

```text
python -m data_creation.transform_social_game_cases [options]
```

## Required Arguments

| Argument | Type | Description |
|--------|-------------|
| `--social-game` | string | Supported values in this release: `beauty_contest`, `escalation_game`. |
| `--input-path` | path | Curated source JSONL file to transform. |
| `--output-dir` | path | Directory for success, failure, skipped, and metadata artifacts. |
| `--few-shot-path` | path | Few-shot JSON asset used for the selected run. |

## Optional Arguments

| Argument | Type | Description |
|--------|-------------|
| `--rubric-path` | path | Shared rubric file. Defaults to repository `transform_rubrics.md`. |
| `--model` | string | OpenAI-compatible chat model identifier. Defaults to `deepseek-chat`. |
| `--num-workers` | integer | Worker count placeholder for row processing. Deterministic behavior is preserved. |
| `--limit` | integer | Process only the first N source rows. |
| `--rerun` | flag | Ignore prior artifacts and rebuild run state from scratch. |
| `--max-retries` | integer | Retry count for row-level transform attempts. |

## Social Game Mapping Contract

| `--social-game` value | Target runtime game | Success artifact | Failure artifact | Skip artifact |
|--------|-------------|-------------|-------------|-------------|
| `beauty_contest` | `Beauty_Contest` | `beauty_contest.success.json` | `beauty_contest.failures.jsonl` | `beauty_contest.skipped.jsonl` |
| `escalation_game` | `Escalation_Game` | `escalation_game.success.json` | `escalation_game.failures.jsonl` | `escalation_game.skipped.jsonl` |

Notes:
- `Diplomacy_Escalation_Game` is out of scope for this contract.
- In this release, `escalation_game` may temporarily reuse the existing Beauty Contest few-shot asset.

## Exit Behavior

| Exit Code | Meaning |
|--------|-------------|
| `0` | Run completed with trustworthy accounting. Some rows may still appear in failure or skipped artifacts. |
| non-zero | Fatal setup or unrecoverable run failure prevented trustworthy completion. |

## Standard Output

The command emits human-readable progress lines and a final summary, for example:

```text
progress=1/10 (10%)
progress=2/10 (20%)
social_game=escalation_game total=10 success=8 failed=1 skipped=1
success_output=/abs/path/.../escalation_game.success.json
failure_output=/abs/path/.../escalation_game.failures.jsonl
metadata_output=/abs/path/.../run_metadata.json
```

## Standard Error

Fatal configuration errors are reported on standard error, including unsupported games, missing prompt assets, invalid CLI inputs, or missing source identity fields during setup.

## Artifact Contract

The command writes:
- one success dataset containing only loadable transformed rows
- one failure artifact containing invalid or unsuccessful rows
- one skipped artifact containing resume-skip rows
- one `run_metadata.json` artifact containing reproducibility and counter data

The command must never mix failure or skipped entries into the main success dataset.

## Validation Rules

- `--social-game` must be explicitly mapped in code; unsupported values fail loudly.
- Each source row must provide non-empty `id` and `source.game_id`, or the row is written to the failure artifact as invalid.
- Every success row must instantiate through the mapped scenario class constructor, `scenario_class(**data)`.
- Deterministic game-specific fields such as canonical `game_name` and payoff data are injected from code/config, not trusted to the model.
- For `escalation_game`:
  - `previous_actions` is optional but authoritative when present.
  - `previous_actions_length` may be used as fallback input when explicit history is absent.
  - if both are present, `previous_actions_length` must equal `len(previous_actions)` or validation fails.

## Compatibility Notes

- This CLI is internal to the repository, but its artifact structure is considered stable for downstream experiments once implemented.
- Adding future social games requires explicit mapping, real scenario-constructor validation, and prompt-asset decisions; no automatic discovery is allowed.
