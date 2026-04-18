---

description: "Task list for Social Game Case Transformation Pipeline"

---

# Tasks: Social Game Case Transformation Pipeline

**Input**: Design documents from `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/`
**Prerequisites**: `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/plan.md`, `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/spec.md`, `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/research.md`, `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/data-model.md`, `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/contracts/social-game-transform-cli.md`

**Tests**: Tests are REQUIRED. Every user story includes failing tests first, integrated validation at the CLI or game-contract boundary, regression pytest, and `mypy` for modified Python code.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or an isolated test slice
- **[Story]**: Maps the task to one user story: `[US1]`, `[US2]`, `[US3]`
- Every task includes an exact absolute file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Refresh the executable test and documentation scaffolds so implementation starts from the current same-game, per-row few-shot design instead of the stale cross-game reuse story.

- [ ] T001 Update the coverage notes in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/features/social_game_transform.feature` for same-game few-shot files, run-present variant filtering, and per-row few-shot packs
- [ ] T002 [P] Refresh the shared source-row and few-shot-example fixtures in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` so they cover multiple `variant_name` values from the same social game
- [ ] T003 [P] Review and tighten the `EscalationGameScenario` contract expectations in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/tests/games/test_escalation_game_config.py` for optional explicit history plus constructor validation

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared selection and audit primitives that all user stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Introduce base-pool, run-matched-pool, and per-row-pack helper structures in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`
- [ ] T005 [P] Add lexical-surface extraction plus greedy weighted `3/4/5`-gram scoring helpers in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`
- [ ] T006 [P] Extend few-shot selection failure stages and audit field shapes in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` so insufficient same-variant or cross-variant pools fail loudly
- [ ] T007 [P] Add foundational helper-level regression coverage for pool derivation and scoring primitives in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py`

**Checkpoint**: Foundation ready. User stories can proceed against stable same-game pool, variant filtering, scoring, and failure primitives.

---

## Phase 3: User Story 1 - Produce Load-Ready Game Cases (Priority: P1) 🎯 MVP

**Goal**: Transform curated `beauty_contest` and plain `escalation_game` source rows into success-only datasets that still load through their real scenario classes after per-row few-shot pack construction is introduced.

**Independent Test**: Run the CLI on small fixtures for both supported social games and confirm that successful rows load through `BeautyContestScenario(**data)` or `EscalationGameScenario(**data)`, while invalid rows and selection failures appear only in failure artifacts.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST and confirm they FAIL before implementation**

- [ ] T008 [P] [US1] Add failing CLI integration coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for same-game few-shot files with success-only datasets across `beauty_contest` and `escalation_game`
- [ ] T009 [P] [US1] Add failing constructor-boundary coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` proving per-row few-shot packs do not weaken `scenario_class(**data)` validation
- [ ] T010 [P] [US1] Add failing escalation history-contract coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/tests/games/test_escalation_game_config.py` for explicit `previous_actions`, fallback `previous_actions_length`, and mismatch rejection
- [ ] T011 [P] [US1] Add failing invalid-identity coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for rows missing top-level `id` or `source.game_id`
- [ ] T012 [P] [US1] Add failing narrative-consistency and style-rule coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for participant naming, behavior-choice wording, and forbidden game-mechanism jargon

### Implementation for User Story 1

- [ ] T013 [US1] Keep the row transformation flow in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` compatible with externally prepared per-row few-shot packs while preserving constructor-boundary validation
- [ ] T014 [US1] Update row transformation and deterministic field injection in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` so transformed rows still produce loadable `Beauty_Contest` and `Escalation_Game` outputs
- [ ] T015 [US1] Implement explicit invalid-identity failure handling in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` for rows missing top-level `id` or `source.game_id`
- [ ] T016 [US1] Implement participant-name consistency, behavior-choice wording consistency, and narrative-style validation in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`
- [ ] T017 [US1] Adjust `EscalationGameScenario` validation behavior in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/games/escalation_game.py` only as needed to keep explicit-history and fallback-length rules aligned with the updated transform flow
- [ ] T018 [US1] Run targeted pytest for `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` and `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/tests/games/test_escalation_game_config.py` covering the US1 loadable-output boundary

**Checkpoint**: User Story 1 delivers loadable success-only datasets for both supported games with the new per-row few-shot flow in place.

---

## Phase 4: User Story 2 - Swap Prompt Assets By Social Game (Priority: P2)

**Goal**: Make prompt assembly explicitly same-game, run-variant-aware, and per-row, starting from the full selected few-shot file and enforcing the exact same-variant-plus-2-cross-variant composition rule.

**Independent Test**: Verify that the CLI builds the base example pool from the selected same-game `--few-shot-path`, filters it to run-present variants only, constructs per-row packs with exactly 2 cross-variant examples, and fails loudly when the eligible pools cannot satisfy the rule.

### Tests for User Story 2 ⚠️

- [ ] T019 [P] [US2] Add failing tests in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for deriving the base example pool directly from the selected same-game `--few-shot-path`
- [ ] T020 [P] [US2] Add failing tests in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for rejecting unsupported social games before transformation begins
- [ ] T021 [P] [US2] Add failing tests in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for filtering to run-present variants only and rejecting examples from absent variants
- [ ] T022 [P] [US2] Add failing tests in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for exact per-row pack composition: 2 cross-variant examples and all remaining examples from the row's own variant
- [ ] T023 [P] [US2] Add failing tests in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for loud failure when same-variant or cross-variant pools are insufficient

### Implementation for User Story 2

- [ ] T024 [US2] Implement base-pool loading and run-variant extraction in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` using the full selected same-game few-shot file as the first pool
- [ ] T025 [US2] Implement explicit unsupported-game rejection in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` at CLI setup time
- [ ] T026 [US2] Implement per-row same-variant and cross-variant pool splitting in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`
- [ ] T027 [US2] Implement deterministic greedy weighted `3/4/5`-gram ranking over `description` and `behavior_choices` in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`
- [ ] T028 [US2] Enforce exact per-row pack composition and explicit setup or row-selection failures in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`
- [ ] T029 [US2] Run targeted pytest for `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` covering same-game file selection, unsupported-game rejection, and per-row pack construction

**Checkpoint**: User Story 2 delivers the new few-shot selection behavior without cross-game fallback or ambiguous pool derivation.

---

## Phase 5: User Story 3 - Resume And Audit A Long Transformation Run (Priority: P3)

**Goal**: Preserve deterministic resume behavior and strengthen auditability by recording candidate, diversity, and few-shot selection evidence alongside the existing success, failure, skipped, and metadata artifacts.

**Independent Test**: Run the CLI twice on partial fixtures and confirm that completed identities are not duplicated, skipped rows are recorded, selection failures are accounted for explicitly, and metadata plus audit artifacts explain how each row was processed.

### Tests for User Story 3 ⚠️

- [ ] T030 [P] [US3] Add failing CLI-output coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for visible progress lines and the final completion summary
- [ ] T031 [P] [US3] Add failing resume and rerun coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for per-row few-shot packs with no duplicate successful rows
- [ ] T032 [P] [US3] Add failing artifact-accounting coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for `*.candidates.jsonl`, `diversity_report.json`, and failure records with `few_shot_selection` stages
- [ ] T033 [P] [US3] Add failing metadata coverage in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` for run-present variants, selected few-shot file provenance, and reconciled success or failed or skipped counters

### Implementation for User Story 3

- [ ] T034 [US3] Implement visible progress lines and final completion-summary output in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`
- [ ] T035 [US3] Extend run metadata and failure-record writing in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` to preserve run-present variants, few-shot file provenance, and explicit `few_shot_selection` failures
- [ ] T036 [US3] Reconcile candidate-artifact, diversity-report, and resume bookkeeping flows in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` so audit outputs stay consistent across reruns
- [ ] T037 [US3] Run targeted pytest for `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py` covering CLI output, resume, rerun, artifact accounting, and metadata reconciliation

**Checkpoint**: All three user stories work independently and the pipeline preserves reproducible audit evidence for long runs.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup, regression validation, documentation, and research-facing evidence.

- [ ] T038 [P] Update CLI usage and artifact documentation in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/README.md` for same-game base pools, per-row packs, visible progress output, and diversity artifacts
- [ ] T039 [P] Refresh executable validation steps in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/quickstart.md` if implementation details changed during TDD
- [ ] T040 Refactor `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` for clarity after all story tests pass, keeping the design simple and explicit
- [ ] T041 [P] Add any missing regression checks for few-shot selection edge cases, narrative-style validation, and audit artifacts in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py`
- [ ] T042 Run `mypy` for `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py` and `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/games/escalation_game.py`
- [ ] T043 Run quickstart validation from `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/quickstart.md` and capture concrete artifact evidence from a same-game few-shot run

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies. Starts immediately.
- **Foundational (Phase 2)**: Depends on Setup completion. Blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion. This is the MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational completion and owns exact same-game pool derivation plus per-row pack-construction rules.
- **User Story 3 (Phase 5)**: Depends on Foundational completion and the success, failure, candidate, diversity, and CLI-output flows exercised by US1 and US2.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Can start right after Foundational. No dependency on later stories.
- **User Story 2 (P2)**: Can start after Foundational, but should own the exact same-game few-shot selection rules so US1 remains focused on loadable outputs.
- **User Story 3 (P3)**: Can start after Foundational, but depends conceptually on the candidate and diversity artifact behavior finalized in US2.

### Within Each User Story

- Tests MUST be written and must fail before implementation.
- Pool derivation comes before pack composition.
- Pack composition comes before final model invocation flow.
- Constructor validation remains the final success boundary.
- Story-level pytest runs complete before moving to the next story.

### Parallel Opportunities

- T002 and T003 can run in parallel.
- T005, T006, and T007 can run in parallel after T004 defines the shared helper shape.
- T008, T009, T010, T011, and T012 can run in parallel.
- T019, T020, T021, T022, and T023 can run in parallel.
- T030, T031, T032, and T033 can run in parallel.
- T038, T039, and T041 can run in parallel once implementation is stable.

---

## Parallel Example: User Story 2

```bash
# Launch the failing selection tests for User Story 2 together:
Task: "T015 [US2] Add failing tests for base-pool loading from --few-shot-path in data_creation/tests/test_transform_social_game_cases.py"
Task: "T016 [US2] Add failing tests for run-present variant filtering in data_creation/tests/test_transform_social_game_cases.py"
Task: "T017 [US2] Add failing tests for exact same-variant-plus-2-cross-variant pack composition in data_creation/tests/test_transform_social_game_cases.py"
Task: "T018 [US2] Add failing tests for insufficient-pool failures in data_creation/tests/test_transform_social_game_cases.py"

# After those tests fail, implement the smallest passing changes:
Task: "T024 [US2] Implement base-pool loading and run-variant extraction in data_creation/transform_social_game_cases.py"
Task: "T026 [US2] Implement per-row pool splitting in data_creation/transform_social_game_cases.py"
Task: "T027 [US2] Implement deterministic greedy weighted 3/4/5-gram ranking in data_creation/transform_social_game_cases.py"
Task: "T028 [US2] Enforce exact per-row pack composition and loud failures in data_creation/transform_social_game_cases.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. **STOP and VALIDATE**: Run the targeted CLI and game-contract tests proving both supported games still produce loadable success rows.
5. Demo the CLI on small fixtures for `beauty_contest` and `escalation_game`.

### Incremental Delivery

1. Build the shared pool and scoring foundation.
2. Deliver loadable per-row transformation for both supported games as the MVP.
3. Add explicit same-game few-shot pool derivation, unsupported-mode rejection, and exact pack composition enforcement.
4. Extend visible progress output plus resume, candidate, diversity, and metadata audit behavior.
5. Finish with docs, `mypy`, quickstart validation, and artifact evidence.

### Parallel Team Strategy

1. One developer stabilizes helper and artifact logic in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/transform_social_game_cases.py`.
2. A second developer writes failing CLI and selection tests in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/data_creation/tests/test_transform_social_game_cases.py`.
3. A third developer maintains `EscalationGameScenario` contract tests in `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/tests/games/test_escalation_game_config.py`.
4. After Foundation is merged, user-story work proceeds with low conflict because the main write scopes are the CLI module, the CLI test module, and the escalation contract test module.

---

## Notes

- `[P]` tasks touch different files or independent test slices.
- Every task includes an exact absolute file path.
- The MVP scope is User Story 1.
- Success datasets must remain success-only throughout implementation.
- Few-shot selection must start from the full selected same-game file contents and must never silently broaden beyond that file.
