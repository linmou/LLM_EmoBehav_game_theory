# Tasks: Social Game Case Transformation Pipeline

**Input**: Design documents from `/specs/002-social-game-transform/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are REQUIRED. Every user story MUST include failing tests first, integrated validation at the relevant system boundary, regression coverage, and mypy for modified Python code.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Data pipeline code lives under `data_creation/`
- Data-pipeline tests live under `data_creation/tests/`
- Game contract validation lives under `tests/games/`
- Feature docs live under `specs/002-social-game-transform/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the feature files and fixture locations used by all stories

- [x] T001 Create the transform feature test specification in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/features/social_game_transform.feature
- [x] T002 Create the transform pipeline test module scaffold in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py
- [x] T003 Create the transform CLI module scaffold in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core shared infrastructure that MUST exist before any user story work

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement shared file-loading, prompt-pack loading, and `.env` credential resolution helpers in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T005 Implement source identity, progress rendering, artifact path, and run-metadata helpers in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T006 Implement shared source-row validation, transformed-row validation, and `BeautyContestScenario` load validation helpers in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Produce Load-Ready Game Cases (Priority: P1) 🎯 MVP

**Goal**: Transform curated `beauty_contest` cases into a success-only dataset that loads through the real Beauty Contest game contract

**Independent Test**: Run the CLI on a small `beauty_contest` fixture and verify that success outputs load as `BeautyContestScenario` instances while invalid rows land only in failure artifacts

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T007 [P] [US1] Add failing CLI contract tests for required arguments, success artifact paths, and summary output in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py
- [x] T008 [P] [US1] Add failing integration tests for successful transformation, success-only output, and failure artifact separation in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py
- [x] T009 [P] [US1] Add failing game-load validation coverage for transformed Beauty Contest rows in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/tests/games/test_beauty_contest_game_config.py

### Implementation for User Story 1

- [x] T010 [US1] Implement `beauty_contest` prompt-pack assembly from /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/transform_rubrics.md and mapped few-shot assets in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T011 [US1] Implement the DeepSeek chat transformation call, response parsing, and structured output normalization in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T012 [US1] Implement success-dataset writing, failure/skipped artifact writing, and final summary emission in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T013 [US1] Implement the CLI entrypoint and end-to-end row processing flow in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T014 [US1] Run targeted pytest coverage for User Story 1 using /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py and /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/tests/games/test_beauty_contest_game_config.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Swap Prompt Assets By Social Game (Priority: P2)

**Goal**: Keep the pipeline reusable by explicit mapping, while V1 still supports only `beauty_contest`

**Independent Test**: Verify that the CLI accepts only explicitly mapped social games, loads the mapped prompt assets for `beauty_contest`, and fails loudly for unsupported games or missing prompt assets

### Tests for User Story 2 ⚠️

- [x] T015 [P] [US2] Add failing tests for explicit social-game mapping, unsupported social-game rejection, and missing prompt-asset handling in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py
- [x] T016 [P] [US2] Add failing tests that prove the system prompt includes both shared rubric content and game-specific few-shot content in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py

### Implementation for User Story 2

- [x] T017 [US2] Implement explicit social-game-to-target mapping and prompt-asset mapping for `beauty_contest` in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T018 [US2] Implement loud rejection for unsupported social games and missing rubric/few-shot assets in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T019 [US2] Implement prompt rendering that composes shared rubric text with mapped few-shot content in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T020 [US2] Run targeted pytest coverage for User Story 2 using /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py

**Checkpoint**: At this point, User Stories 1 and 2 should both work independently

---

## Phase 5: User Story 3 - Resume And Audit A Long Transformation Run (Priority: P3)

**Goal**: Provide resumability, deterministic row accounting, and reproducible run metadata for long-running transformations

**Independent Test**: Run the CLI twice on a partial fixture and confirm that `id + source.game_id` prevents duplicate successes, preserves prior failure records, and writes reconciled run metadata

### Tests for User Story 3 ⚠️

- [x] T021 [P] [US3] Add failing tests for resume behavior, duplicate prevention, and `--rerun` override handling in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py
- [x] T022 [P] [US3] Add failing tests for run-metadata counters, provenance retention, and terminal-state bookkeeping in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py

### Implementation for User Story 3

- [x] T023 [US3] Implement resume-state loading and identity-based skip logic using `id + source.game_id` in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T024 [US3] Implement deterministic failure/skipped terminal records and `--rerun` behavior in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T025 [US3] Implement run-metadata recording, counter reconciliation, and provenance persistence in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T026 [US3] Run targeted pytest coverage for User Story 3 using /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, docs, and research-evidence checks across all stories

- [x] T027 [P] Update quick usage and artifact expectations in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/README.md
- [x] T028 Refactor /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py for clarity after all story tests pass
- [x] T029 Run mypy for /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T030 Run quickstart validation from /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/quickstart.md and capture artifact evidence
- [x] T031 Verify progress output, run metadata, and example artifact contents from /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/quickstart.md against the implemented CLI

---

## Phase 7: Contract-Validation Alignment

**Purpose**: Align the implementation with the clarified rule that the real game scenario constructor is the contract boundary

- [x] T032 [P] Add failing validation-alignment coverage in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py
- [x] T033 [P] Add failing game-contract regression coverage in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/tests/games/test_beauty_contest_game_config.py
- [x] T034 Simplify transformed-row validation to rely on direct scenario construction in /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py
- [x] T035 Run targeted pytest coverage for the contract-alignment change using /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py and /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/tests/games/test_beauty_contest_game_config.py
- [x] T036 Run mypy for /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py after the refactor

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; start immediately
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion; this is the MVP
- **User Story 2 (Phase 4)**: Depends on Foundational completion and reuses the CLI skeleton from User Story 1
- **User Story 3 (Phase 5)**: Depends on Foundational completion and the artifact flow created in User Story 1
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational; no dependency on later stories
- **User Story 2 (P2)**: Can start after Foundational, but is most efficient after User Story 1 establishes the base CLI path
- **User Story 3 (P3)**: Can start after Foundational, but depends conceptually on the success/failure artifact model from User Story 1

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Shared validations and helpers from Foundational phase come before story-specific implementation
- CLI behavior comes after prompt/mapping logic and row-validation logic
- Story-level pytest verification comes before moving to the next story

### Parallel Opportunities

- T001 and T002 can run in parallel
- T007, T008, and T009 can run in parallel
- T015 and T016 can run in parallel
- T021 and T022 can run in parallel
- T027 and T031 can run in parallel once implementation is complete

---

## Parallel Example: User Story 1

```bash
# Launch all User Story 1 test writing tasks together:
Task: "T007 [US1] Add CLI contract tests in data_creation/tests/test_transform_social_game_cases.py"
Task: "T008 [US1] Add integration tests in data_creation/tests/test_transform_social_game_cases.py"
Task: "T009 [US1] Add game-load validation tests in tests/games/test_beauty_contest_game_config.py"

# After those fail, implement the core User Story 1 work in sequence:
Task: "T010 [US1] Implement prompt-pack assembly in data_creation/transform_social_game_cases.py"
Task: "T011 [US1] Implement DeepSeek transformation call in data_creation/transform_social_game_cases.py"
Task: "T012 [US1] Implement artifact writing in data_creation/transform_social_game_cases.py"
Task: "T013 [US1] Implement CLI entrypoint in data_creation/transform_social_game_cases.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run the User Story 1 pytest targets and confirm success outputs load through `BeautyContestScenario`
5. Demo the CLI on a small `beauty_contest` fixture

### Incremental Delivery

1. Build the shared CLI and validation foundation
2. Deliver `beauty_contest` transformation and success-only artifacts as MVP
3. Add explicit prompt-pack mapping and unsupported-game rejection
4. Add resume, rerun, and run-metadata accounting
5. Finish with docs, mypy, and quickstart validation

### Parallel Team Strategy

With multiple developers:

1. One developer handles Phase 2 shared helpers in `data_creation/transform_social_game_cases.py`
2. A second developer writes the failing integration tests in `data_creation/tests/test_transform_social_game_cases.py`
3. A third developer extends `tests/games/test_beauty_contest_game_config.py` for real loader validation
4. After Foundation is merged, story work can proceed with low conflict because most tasks center on one CLI module and one test module

---

## Notes

- [P] tasks touch different files or independent test slices
- Every task includes an exact file path
- The MVP scope is User Story 1 only
- The main dataset must remain success-only throughout implementation
- Avoid adding a service layer, generalized game registry, or fallback identity heuristics in this feature
