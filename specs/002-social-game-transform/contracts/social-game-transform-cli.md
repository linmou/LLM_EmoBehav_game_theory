# Contract: Social Game Transform CLI

## Purpose

Define the user-facing CLI contract for transforming curated social-game source rows into success-only datasets loadable by the repository's real game scenario classes, with auditable few-shot selection and diversity artifacts.

## Command

```text
python -m data_creation.transform_social_game_cases [options]
```

## Required Arguments

| Argument | Type | Description |
|--------|-------------|
| `--social-game` | string | Supported values in this release: `beauty_contest`, `escalation_game`. |
| `--input-path` | path | Curated source JSONL file to transform. |
| `--output-dir` | path | Directory for success, failure, skipped, candidate, diversity, and metadata artifacts. |
| `--few-shot-path` | path | Same-game few-shot example library used to build per-row few-shot packs. |

## Optional Arguments

| Argument | Type | Description |
|--------|-------------|
| `--rubric-path` | path | Shared rubric file. Defaults to `data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md`. |
| `--model` | string | OpenAI-compatible chat model identifier. Defaults to `deepseek-chat`. |
| `--num-workers` | integer | Worker count for row processing. |
| `--limit` | integer | Process only the first N source rows. |
| `--rerun` | flag | Ignore prior artifacts and rebuild run state from scratch. |
| `--max-retries` | integer | Retry count for row-level transform attempts. |
| `--num-candidates` | integer | Number of model candidates to generate per row before final row selection. |
| `--temperature` | float | Temperature passed to candidate generation. |

## Social Game Mapping Contract

| `--social-game` value | Target runtime game | Success artifact | Failure artifact | Skip artifact | Candidate artifact |
|--------|-------------|-------------|-------------|-------------|-------------|
| `beauty_contest` | `Beauty_Contest` | `beauty_contest.success.json` | `beauty_contest.failures.jsonl` | `beauty_contest.skipped.jsonl` | `beauty_contest.candidates.jsonl` |
| `escalation_game` | `Escalation_Game` | `escalation_game.success.json` | `escalation_game.failures.jsonl` | `escalation_game.skipped.jsonl` | `escalation_game.candidates.jsonl` |

Notes:
- `Diplomacy_Escalation_Game` is out of scope for this contract.
- The few-shot library must belong to the same social game as the run.

## Few-Shot Selection Contract

- The first example pool is the full contents of the selected same-game `--few-shot-path` file.
- The run derives the set of variants present in the selected input rows.
- Only examples from that file whose variant appears in that run-present set may enter the eligible pool.
- The command builds a separate few-shot pack for each source row.
- Each per-row pack must contain at least 1 example from the source row's own variant.
- For multi-variant runs, each per-row pack must contain exactly 2 examples from other run-present variants.
- Every remaining few-shot example in the pack must come from the source row's own variant.
- Lexical diversity scoring for few-shot ranking uses only `description` and `behavior_choices`.
- Ranking follows a deterministic greedy weighted 3/4/5-gram gain rule.
- If the eligible pool cannot satisfy the same-variant rule, or the multi-variant cross-variant rule, the system fails loudly rather than broadening the pool.

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

Fatal configuration errors are reported on standard error, including unsupported games, missing prompt assets, invalid CLI inputs, missing source identity fields during setup, or insufficient eligible few-shot pools.

## Artifact Contract

The command writes:
- one success dataset containing only loadable transformed rows
- one failure artifact containing invalid or unsuccessful rows
- one skipped artifact containing resume-skip rows
- one candidate artifact containing generated pre-selection rows
- one `diversity_report.json` artifact containing classic n-gram metrics
- one `run_metadata.json` artifact containing reproducibility and counter data

The command must never mix failure or skipped entries into the main success dataset.
For game-specific runtime fields such as `payoff_matrix`, the written success artifact must remain directly reloadable through the mapped scenario class from the saved JSON payload, not only during in-memory pre-write validation.

## Validation Rules

- `--social-game` must be explicitly mapped in code; unsupported values fail loudly.
- Each source row must provide non-empty `id` and `source.game_id`, or the row is written to the failure artifact as invalid.
- Every success row must instantiate through the mapped scenario class constructor, `scenario_class(**data)`.
- Deterministic game-specific fields such as canonical `game_name` and payoff data are injected from code/config, not trusted to the model.
- For `escalation_game`:
  - `previous_actions` is optional but authoritative when present.
  - `previous_actions_length` may be used as fallback input when explicit history is absent.
  - If both are present, `previous_actions_length` must equal `len(previous_actions)` or validation fails.

## Compatibility Notes

- This CLI is internal to the repository, but its artifact structure is considered stable for downstream experiments once implemented.
- Adding future social games requires explicit mapping, real scenario-constructor validation, and dedicated same-game prompt assets; no automatic discovery is allowed.
