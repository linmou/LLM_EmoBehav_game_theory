# Feature Specification: Social Game Case Transformation Pipeline

**Feature Branch**: `002-social-game-transform`  
**Created**: 2026-04-06  
**Status**: Draft  
**Input**: User description: "build a pipeline that use transform structuralized data from /home/jjl7137/diplomacy_cicero/social_game_outputs/<social_game>/curated_cases/<social_game>_cases.jsonl into cases can be loaded by game class under ./games (e.g. games/beauty_contest.py ) . use deepseek-chat model to transform , api is in .env, you can refer to .env.example , for the prompt , use  transform_rubrics.md as part of the system prompt and pluggable few-shot prompt   beauty_contest_few_shot_examples.json ( can be loaded from other files)"

## Clarifications

### Session 2026-04-06

- Q: What is the intended V1 support scope for social games? → A: Start with beauty_contest only, but leave extension space for any social game that has an explicit mapping and prompt assets.
- Q: What identity rule should the pipeline use for resume and deduplication? → A: Use `id + source.game_id`, and keep source metadata as provenance.
- Q: How should the run behave when some source records are invalid? → A: Continue processing valid records and write invalid ones to failure artifacts.
- Q: What should the main transformed dataset contain? → A: Only successful transformed cases; failures and skips live in separate machine-readable artifacts.
- Q: What is the target-game contract validation rule? → A: Use the real game scenario class constructor, `scenario_class(**data)`, as the contract validation.

### Session 2026-04-17

- Q: What is the first-release support scope now that the current implementation is hardcoded for beauty_contest? → A: Support both `beauty_contest` and `escalation_game` in the first release.
- Q: How should the pipeline source few-shot examples for `escalation_game` in this release? → A: Reuse the existing `beauty_contest` few-shot asset temporarily.
- Q: How should game-specific contract fields be produced for supported games in this release? → A: The pipeline fills game-specific contract fields in code from explicit mapping/config.
- Q: How should `previous_actions_length` be set for `escalation_game` outputs in this release? → A: Let the model generate `previous_actions_length` from the source row.
- Q: Which escalation-game contract should this feature target in the first release? → A: Support only the plain `Escalation_Game` contract in this feature.
- Q: When both `previous_actions` and `previous_actions_length` are present for `escalation_game`, which field is authoritative? → A: `previous_actions` is authoritative, and a mismatched `previous_actions_length` is a validation error.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Produce Load-Ready Game Cases (Priority: P1)

A researcher selects a supported social game and transforms every curated source case into scenario records that can be consumed by the corresponding game definition without manual cleanup or field renaming. In the first release, the supported social games are `beauty_contest` and `escalation_game`.

**Why this priority**: The pipeline is only useful if it closes the gap between curated source data and the scenario schema already used by experiments.

**Independent Test**: Can be fully tested by running the pipeline for each first-release supported social game and confirming that successful outputs load directly into the target game scenario class.

**Acceptance Scenarios**:

1. **Given** a curated source file for a supported social game, **When** the researcher runs the transformation pipeline, **Then** the pipeline produces a transformed dataset whose successful records match the target game's accepted scenario shape.
2. **Given** a transformed record marked as successful, **When** the researcher loads it through the existing game workflow, **Then** the record is accepted without manual editing.
3. **Given** a run where some source rows are invalid, **When** the pipeline finishes, **Then** valid rows remain available in the transformed dataset and invalid rows are recorded separately with failure details.
4. **Given** a completed run, **When** a downstream experiment reads the main transformed dataset, **Then** it encounters only loadable successful cases and no failure entries.

---

### User Story 2 - Swap Prompt Assets By Social Game (Priority: P2)

A researcher can reuse the same pipeline across different social games by pairing the shared rubric instructions with a game-specific example pack, rather than rewriting the transformation workflow each time. A social game becomes eligible only after its target-game mapping and prompt assets are defined explicitly.

**Why this priority**: The repo already holds multiple game families, so the pipeline must be reusable instead of hard-coded around one narrow case.

**Independent Test**: Can be fully tested by configuring two supported social games with different example assets and confirming that both run through the same workflow while preserving each game's scenario constraints.

**Acceptance Scenarios**:

1. **Given** a supported social game with a shared rubric asset and its own example asset, **When** the researcher runs the pipeline, **Then** the pipeline applies both instruction sources during transformation.
2. **Given** a second supported social game, **When** the researcher switches the selected social game, **Then** the pipeline reuses the same workflow and enforces the second game's output contract.
3. **Given** first-release support for `escalation_game`, **When** the researcher runs the pipeline for that game, **Then** the pipeline may temporarily reuse the existing `beauty_contest` few-shot asset while still enforcing the `escalation_game` scenario contract.

---

### User Story 3 - Resume And Audit A Long Transformation Run (Priority: P3)

A researcher can interrupt a long run and resume it later while keeping a complete accounting of which source records succeeded, failed, or were already completed.

**Why this priority**: These case-building jobs can be large, expensive, and sensitive to external model failures, so restartability and traceability are part of the feature, not optional polish.

**Independent Test**: Can be fully tested by interrupting a partial run, restarting it, and confirming that completed records are not duplicated and all source rows remain accounted for.

**Acceptance Scenarios**:

1. **Given** a partially completed run, **When** the researcher restarts the same job, **Then** the pipeline resumes from unfinished records and preserves already completed outputs.
2. **Given** a run containing both successful and failed records, **When** the run completes, **Then** the researcher can review machine-readable output showing the status of every source record.

### Edge Cases

- What happens when a source row is malformed JSON or lacks the metadata needed to identify its origin?
- What happens when a source row is missing either the top-level record identifier or `source.game_id`, which together define the resume and deduplication identity?
- What happens when the transformed output is missing required scenario fields, required choice keys, or valid prior-round action structure for a multi-turn game?
- What happens when an `escalation_game` transformed row provides both `previous_actions` and `previous_actions_length`, but the length does not equal the number of explicit prior actions?
- How does the system handle a supported social game whose example asset is missing or whose target game mapping is undefined, except for the temporary first-release allowance that `escalation_game` may reuse the existing `beauty_contest` few-shot asset?
- How does the system handle a restart after interruption when some outputs were already written and some failure records already exist?
- How does the system prevent a rerun from duplicating already transformed successes while still preserving prior failure records for invalid rows?
- What happens when the transformed narrative uses inconsistent participant names across `description`, `participants`, and prior-round action entries?
- What happens when the transformation output violates the approved narrative rules, such as explicit game-mechanism jargon or fabricated prior-round detail?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a selected social game and process all records from that social game's curated source case file.
- **FR-001a**: System MUST support both `beauty_contest` and `escalation_game` in the first release.
- **FR-001b**: For `escalation_game`, the first release MUST target only the plain `Escalation_Game` contract and MUST NOT include `Diplomacy_Escalation_Game`.
- **FR-002**: System MUST transform each successful source record into a scenario record that matches the target game's accepted scenario fields and choice structure.
- **FR-003**: System MUST validate each transformed record by constructing the target game scenario class with `scenario_class(**data)` before the record is added to the final transformed dataset.
- **FR-004**: System MUST mark an individual record as failed, rather than silently coercing it, when required fields, choice keys, participant data, or prior-round action structures are missing or invalid.
- **FR-004a**: System MUST continue processing remaining source records after an individual record fails validation or transformation.
- **FR-005**: System MUST build transformation instructions from one shared rubric asset plus one social-game-specific example asset for each run.
- **FR-006**: System MUST allow the social-game-specific example asset to be changed independently of the shared rubric asset and core transformation workflow.
- **FR-006a**: System MUST treat any social game other than `beauty_contest` and `escalation_game` as unsupported until that social game has an explicit target-game mapping and its required prompt assets.
- **FR-006b**: System MUST allow `escalation_game` to reuse the existing `beauty_contest` few-shot asset temporarily in the first release, while still requiring an explicit target-game mapping and contract validation against the `escalation_game` scenario class.
- **FR-007**: System MUST preserve source traceability for every successful, failed, or skipped record by storing the source identifier or enough source metadata to recover the originating row.
- **FR-007a**: System MUST use the combination of the top-level record `id` and `source.game_id` as the identity for resume and deduplication.
- **FR-007b**: System MUST treat a source record as invalid when either the top-level record `id` or `source.game_id` is missing.
- **FR-008**: System MUST provide visible run progress and a completion summary containing transformed, failed, skipped, and total record counts.
- **FR-009**: System MUST support resume after interruption by recognizing already completed records and continuing from the first unfinished record unless the operator explicitly requests a full rerun.
- **FR-010**: System MUST write machine-readable failure artifacts that capture the source record reference, the failing stage, and a diagnostic message for each unsuccessful transformation.
- **FR-010a**: System MUST preserve successful transformed records in the same run even when other records are written to failure artifacts.
- **FR-010b**: System MUST keep the main transformed dataset limited to successful transformed cases.
- **FR-010c**: System MUST write failed and skipped records to separate machine-readable artifacts rather than mixing them into the main transformed dataset.
- **FR-011**: System MUST keep participant naming, behavior-choice wording, and prior-round action descriptions internally consistent within each transformed scenario.
- **FR-011a**: For `escalation_game`, `previous_actions` MUST be treated as the primary prior-history representation when present. `previous_actions` is optional, but when supplied it MUST drive validation and downstream scenario construction ahead of `previous_actions_length`.
- **FR-011b**: For `escalation_game`, `previous_actions_length` MAY still be supplied as a simpler fallback representation when explicit `previous_actions` are unavailable.
- **FR-011c**: When an `escalation_game` transformed row supplies both `previous_actions` and `previous_actions_length`, the system MUST require `previous_actions_length == len(previous_actions)`; otherwise the row MUST fail validation rather than being silently coerced.
- **FR-012**: System MUST ensure transformed narratives follow the approved style constraints for historical framing, stakeholder tradeoff explanation, and avoidance of explicit game-mechanism jargon.
- **FR-013**: System MUST produce transformed artifacts that existing game-loading workflows can consume without manual field renaming or hand editing.
- **FR-013a**: System MUST treat successful output as contract-valid only when the corresponding target game scenario class accepts the transformed record through direct construction.
- **FR-013b**: System MUST inject deterministic game-specific contract fields from explicit pipeline mapping/config rather than relying on the model to invent them. This includes fields such as canonical game name, payoff data, and other required per-game structural defaults.
- **FR-013c**: For `escalation_game`, the system MUST let the transformed output carry a model-generated `previous_actions_length` derived from the source row, while still validating the final record through the real scenario constructor.
- **FR-014**: System MUST record the output dataset location and run metadata needed for downstream experiments to reproduce the transformation run.

### Key Entities *(include if feature involves data)*

- **Curated Social Game Case**: A structured source record from a selected social game's curated case dataset, including its source metadata, labels, metrics, event history, and the identity pair formed by the top-level `id` and `source.game_id`.
- **Transformation Prompt Pack**: The instruction bundle used for one run, composed of a shared rubric asset and a selected social-game-specific example asset.
- **Transformed Game Scenario Case**: A normalized scenario record intended to satisfy the corresponding target game's scenario contract, including scenario text, participant list, behavior choices, optional prior-round actions, and provenance metadata.
- **Escalation Game History Representation**: For `escalation_game`, prior history may be represented either by optional explicit `previous_actions` or by fallback `previous_actions_length`. When both are present, explicit `previous_actions` is canonical and the fallback length must match it.
- **Transformation Run Record**: The audit trail for one pipeline execution, including run parameters, progress counters, success/failure/skipped status for each source record, output locations, and diagnostic messages.
- **Success Dataset**: The main transformed-case artifact containing only records that passed transformation and target-game validation.
- **Failure Artifact Set**: Separate machine-readable artifacts that store failed and skipped source records with their identity, stage, and diagnostic details.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of transformed records marked as successful can be loaded by direct target-game scenario construction, `scenario_class(**data)`, without manual field edits.
- **SC-002**: 100% of source rows in a run are accounted for as transformed, failed, or skipped, with no untracked records.
- **SC-002a**: In a mixed-quality run, valid rows remain present in the transformed dataset while 100% of invalid rows appear in failure artifacts.
- **SC-002b**: 100% of records present in the main transformed dataset are loadable successful cases.
- **SC-003**: After an interrupted run is restarted, previously completed records are not duplicated in the final transformed dataset.
- **SC-004**: A researcher can switch between `beauty_contest` and `escalation_game` by changing the selected game inputs, while keeping the same transformation workflow and run-accounting behavior.
- **SC-005**: For any failed record in a run, a researcher can identify the originating source row and failure reason from the generated artifacts in under 2 minutes.

## Assumptions

- The first release scope includes `beauty_contest` and `escalation_game`.
- In the first release, `escalation_game` may temporarily reuse the existing `beauty_contest` few-shot asset.
- Additional social games will be added only when each one has a corresponding target scenario contract, explicit mapping rules, and required prompt assets.
- `Diplomacy_Escalation_Game` is out of scope for this feature and should be handled as a separate follow-up if needed.
- Source records provide both a top-level `id` and `source.game_id`, and that pair is stable enough to support resume, deduplication, and traceability.
- The first delivery will rely on one approved external transformation provider configured through the repository environment, with no fallback provider or silent degradation path.
- Social-game-specific example assets are curated separately from the pipeline itself and are treated as required inputs for supported games.
- The transformed dataset will be stored in a repo location that downstream experiments can reference directly as a scenario data source.
