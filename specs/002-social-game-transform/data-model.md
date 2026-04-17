# Data Model: Social Game Case Transformation Pipeline

## Curated Social Game Case

Purpose: Represents one source row from a curated social-game JSONL dataset selected for transformation.

Fields:
- `id`: string. Curated sample identifier.
- `source.game_id`: string. Upstream source game identifier.
- `source.dataset`: string | null. Source dataset label.
- `source.line_number`: integer | null. Source file provenance.
- `episode_type`: string | null. Mechanism-family label from the curated data.
- `variant_name`: string | null. Variant label from the curated data.
- `players`: list[string] | null. Source participant labels.
- `phases`: list[string] | null. Source phase markers.
- `events`: list[object] | null. Structured event history used in the prompt.
- `labels`: object | null. Curated annotations.
- `metrics`: object | null. Curated numeric summaries.

Validation rules:
- `id` is required and non-empty.
- `source.game_id` is required and non-empty.
- The deterministic resume identity is `id + source.game_id`.

## Prompt Pack

Purpose: Defines the prompt assets and runtime target mapping for one selected social game.

Fields:
- `social_game`: string. Supported values in this release: `beauty_contest`, `escalation_game`.
- `rubric_path`: path. Shared rubric file.
- `few_shot_path`: path. Selected few-shot asset.
- `few_shot_examples`: list[object]. Parsed few-shot payload.
- `target_game_name`: string. Canonical runtime game name, for example `Beauty_Contest` or `Escalation_Game`.
- `scenario_class`: Python scenario class reference. The real contract validator for success rows.
- `artifact_names`: object. Success/failure/skip output filenames.

Validation rules:
- `social_game` must map explicitly in code; unsupported values fail fast.
- `rubric_path` must exist.
- `few_shot_path` must exist.
- `few_shot_examples` must parse as a JSON list.
- `escalation_game` may temporarily reuse the Beauty Contest few-shot asset in this release.

## Transformation Request

Purpose: Immutable inputs used for one row transformation attempt.

Fields:
- `identity_key`: string. `id + source.game_id`.
- `source_case`: Curated Social Game Case.
- `prompt_pack`: Prompt Pack.
- `model_name`: string. Chat model identifier.
- `retry_index`: integer. Attempt number for the row.

Validation rules:
- `identity_key` must be unique within a run.
- `retry_index` starts at `0` and increases only for row-level retries.

## Beauty Contest Scenario Row

Purpose: Normalized output row intended to instantiate through `BeautyContestScenario`.

Fields:
- `scenario`: string.
- `description`: string.
- `participants`: list[object] with participant `name`.
- `behavior_choices`: object with `commit_0`, `commit_1`, `commit_2`, `commit_3`.
- `previous_actions`: list[object]. Optional round-history records accepted by `BeautyContestScenario`.
- `game_name`: string. Canonical value `Beauty_Contest`.
- `game_category`: string | null. Optional mechanism label preserved from the transform.
- `provenance`: object with source identifiers and source-location metadata.
- `payoff_matrix`: object. Injected deterministic field, currently empty/default-compatible for this game.

Validation rules:
- Must instantiate successfully as `BeautyContestScenario`.
- If `previous_actions` is present, every entry must satisfy that scenario class’s round-history rules.

## Escalation Game Scenario Row

Purpose: Normalized output row intended to instantiate through `EscalationGameScenario`.

Fields:
- `scenario`: string.
- `description`: string.
- `participants`: list[object] with participant `name`.
- `behavior_choices`: object with `escalate`, `withdraw`.
- `previous_actions`: optional explicit prior-history representation.
- `previous_actions_length`: optional integer fallback history representation.
- `game_name`: string. Canonical value `Escalation_Game`.
- `provenance`: object with source identifiers and source-location metadata.
- `payoff_matrix`: object. Injected deterministic escalation payoff matrix.

Validation rules:
- Must instantiate successfully as `EscalationGameScenario`.
- `previous_actions` is optional.
- `previous_actions_length` is optional.
- If `previous_actions` is present, it is the canonical history representation.
- If both `previous_actions` and `previous_actions_length` are present, then `previous_actions_length` must equal `len(previous_actions)`.
- If explicit `previous_actions` is absent, `previous_actions_length` may be used as fallback input.

## Transformed Game Scenario Case

Purpose: Union of the per-game success-row shapes written to the success dataset.

Variants:
- Beauty Contest Scenario Row
- Escalation Game Scenario Row

Shared rules:
- Every success row carries provenance for the source identity pair.
- Every success row must pass direct scenario-constructor validation for the mapped target game.
- Deterministic structural fields are injected from code/config, not trusted to the model.

## Failure Record

Purpose: Captures one unsuccessful transformation attempt or invalid source row.

Fields:
- `identity_key`: string | null.
- `id`: string | null.
- `source_game_id`: string | null.
- `stage`: string. Typical values include `input_validation`, `transform`, `scenario_load`.
- `error_type`: string.
- `message`: string.
- `source_snapshot`: object. Minimal diagnostic subset of the source row.
- `timestamp`: string. UTC ISO timestamp.

Validation rules:
- Failure records are machine-readable.
- Invalid or unsuccessful rows must not enter the success dataset.
- Every finalized failed row remains traceable to the source identity when available.

## Skipped Record

Purpose: Captures one row skipped due to resume bookkeeping.

Fields:
- `identity_key`: string.
- `stage`: string. Canonical value `resume_skip`.
- `message`: string.
- `timestamp`: string. UTC ISO timestamp.

Validation rules:
- Skipped rows are written only to the skipped artifact, never to the success dataset.

## Run Metadata

Purpose: Stores run-level reproducibility and bookkeeping details.

Fields:
- `run_id`: string.
- `social_game`: string.
- `input_path`: path.
- `success_output_path`: path.
- `failure_output_path`: path.
- `skip_output_path`: path.
- `model_name`: string.
- `rubric_path`: path.
- `few_shot_path`: path.
- `completed_identities`: list[string].
- `counts.total`: integer.
- `counts.success`: integer.
- `counts.failed`: integer.
- `counts.skipped`: integer.

Validation rules:
- Count totals must reconcile to processed rows.
- `completed_identities` must reflect terminal success/failure rows used for resume decisions.

## State Transitions

```text
Source Row
  -> Invalid Source Row
  -> Queued For Transformation
      -> Transform Failure
      -> Scenario Contract Failure
      -> Success Dataset Entry

Prior Artifact + Matching Identity
  -> Resume Skip Entry
```

Rules:
- A processed row reaches exactly one terminal artifact state in a finalized run.
- Resume logic uses terminal identities from prior success and failure artifacts plus run metadata.
