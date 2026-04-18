# Data Model: Social Game Case Transformation Pipeline

## Curated Social Game Case

Purpose: Represents one source row from a curated social-game JSONL dataset selected for transformation.

Fields:
- `id`: string. Curated sample identifier.
- `source.game_id`: string. Upstream source game identifier.
- `source.dataset`: string | null. Source dataset label.
- `source.line_number`: integer | null. Source file provenance.
- `episode_type`: string | null. Mechanism-family label from the curated data.
- `variant_name`: string | null. Variant label used to determine row-level few-shot eligibility.
- `players`: list[string] | null. Source participant labels.
- `phases`: list[string] | null. Source phase markers.
- `events`: list[object] | null. Structured event history used in the prompt.
- `labels`: object | null. Curated annotations.
- `metrics`: object | null. Curated numeric summaries.

Validation rules:
- `id` is required and non-empty.
- `source.game_id` is required and non-empty.
- The deterministic resume identity is `id + source.game_id`.
- The row must expose enough variant metadata for few-shot pack construction.

## Same-Game Few-Shot Library

Purpose: The full few-shot asset loaded for the selected social game before run-level filtering.

Fields:
- `social_game`: string. Supported values in this release: `beauty_contest`, `escalation_game`.
- `few_shot_path`: path. JSON file containing all curated few-shot examples for that social game.
- `examples`: list[object]. Parsed few-shot examples.

Validation rules:
- The file must exist.
- The parsed payload must be a JSON list.
- Every example must carry a variant label that can be matched against run-present variants.
- This library is the base example pool used before any run-level or row-level filtering.

## Run-Matched Few-Shot Pool

Purpose: The subset of the same-game few-shot library eligible for the current run.

Fields:
- `social_game`: string.
- `run_variants`: list[string]. Unique variants found in the current run input.
- `eligible_examples`: list[object]. Same-game few-shot examples whose variant appears in `run_variants`.
- `examples_by_variant`: object. Eligible examples indexed by variant.

Validation rules:
- All eligible examples come from the selected social game.
- No example whose variant is absent from `run_variants` may enter the eligible pool.
- The eligible pool is derived only by filtering the base example pool loaded from the selected `few_shot_path`.
- Run setup fails if no eligible examples exist for any run-present variant.

## Few-Shot Lexical Surface

Purpose: The text surface used to score few-shot diversity.

Fields:
- `description_text`: string.
- `behavior_choice_text`: string. Flattened text from all behavior-choice values.

Validation rules:
- Diversity scoring uses only `description_text` and `behavior_choice_text`.
- Other serialized fields are excluded from lexical scoring.

## Per-Row Few-Shot Pack

Purpose: The few-shot example set constructed for one source row.

Fields:
- `identity_key`: string. `id + source.game_id`.
- `row_variant`: string.
- `same_variant_examples`: list[object]. Selected examples whose variant matches `row_variant`.
- `cross_variant_examples`: list[object]. Selected examples from other run-present variants.
- `selected_examples`: list[object]. Final ordered pack passed into prompt assembly.
- `cross_variant_count`: integer. Must be `2`.

Validation rules:
- The pack is built only from the run-matched few-shot pool.
- `cross_variant_count` must equal `2`.
- Every selected example other than the two cross-variant examples must come from `row_variant`.
- Pack construction fails loudly if there are not enough same-variant or cross-variant eligible examples.

## Few-Shot Selection Score

Purpose: Records the deterministic ranking state used while constructing a per-row pack.

Fields:
- `candidate_example_id`: string | null.
- `new_3grams`: integer.
- `new_4grams`: integer.
- `new_5grams`: integer.
- `overlap_3grams`: integer.
- `overlap_4grams`: integer.
- `overlap_5grams`: integer.
- `total_score`: integer.

Validation rules:
- Scores are computed from the candidate example's lexical surface only.
- Ranking follows a greedy weighted n-gram gain rule.
- Re-running the same pool and same rule yields the same selected order.

## Prompt Pack

Purpose: Defines the prompt assets and runtime target mapping used when transforming one source row.

Fields:
- `social_game`: string. Supported values in this release: `beauty_contest`, `escalation_game`.
- `rubric_path`: path. Shared rubric file.
- `rubric_text`: string.
- `target_game_name`: string. Canonical runtime game name, for example `Beauty_Contest` or `Escalation_Game`.
- `scenario_class`: Python scenario class reference. Real contract validator for success rows.
- `payoff_matrix`: object. Deterministic runtime payoff data from `games.game_configs`.
- `few_shot_pack`: Per-Row Few-Shot Pack.

Validation rules:
- `social_game` must map explicitly in code; unsupported values fail fast.
- `rubric_path` must exist.
- `few_shot_pack` must satisfy the per-row composition rules before prompt assembly.

## Transformation Request

Purpose: Immutable inputs used for one row transformation attempt.

Fields:
- `identity_key`: string. `id + source.game_id`.
- `source_case`: Curated Social Game Case.
- `prompt_pack`: Prompt Pack.
- `model_name`: string. Chat model identifier.
- `retry_index`: integer. Attempt number for the row.
- `temperature`: float.

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
- `payoff_matrix`: object. Injected deterministic field.

Validation rules:
- Must instantiate successfully as `BeautyContestScenario`.
- If `previous_actions` is present, every entry must satisfy that scenario class's round-history rules.

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

## Candidate Record

Purpose: Captures one generated candidate output before final row selection.

Fields:
- `identity_key`: string.
- `candidate_index`: integer.
- `transformed_row`: object.
- `selection_score`: number | null.

Validation rules:
- Candidate artifacts are audit-only and must not weaken the success-only dataset contract.
- Candidate records remain traceable to the source identity.

## Diversity Report

Purpose: Summarizes classic lexical diversity metrics for selected outputs and supporting artifacts.

Fields:
- `description_count`: integer.
- `selected_description_metrics.distinct_1`: float.
- `selected_description_metrics.distinct_2`: float.
- `selected_description_metrics.distinct_3`: float.
- `repeated_3grams`: list[object].
- `candidate_counts.generated`: integer.
- `candidate_counts.selected`: integer.

Validation rules:
- Metrics are computed from selected descriptions using classic n-gram distinctness.
- Diversity reports are supplemental artifacts and do not replace constructor validation.

## Failure Record

Purpose: Captures one unsuccessful transformation attempt or invalid source row.

Fields:
- `identity_key`: string | null.
- `id`: string | null.
- `source_game_id`: string | null.
- `stage`: string. Typical values include `input_validation`, `few_shot_selection`, `transform`, `scenario_load`.
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
- `candidate_output_path`: path.
- `diversity_output_path`: path.
- `model_name`: string.
- `rubric_path`: path.
- `few_shot_path`: path.
- `run_variants`: list[string].
- `completed_identities`: list[string].
- `counts.total`: integer.
- `counts.success`: integer.
- `counts.failed`: integer.
- `counts.skipped`: integer.

Validation rules:
- Count totals must reconcile to processed rows.
- `completed_identities` must reflect terminal success/failure rows used for resume decisions.
- Metadata must preserve enough information to reconstruct the same run-level few-shot eligibility pool.

## State Transitions

```text
Source Row
  -> Invalid Source Row
  -> Few-Shot Pack Construction Failure
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
