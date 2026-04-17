---

description: "Task list for Social Game Case Transformation Pipeline"

---

# Tasks: Social Game Case Transformation Pipeline

**Input**: Design documents from `/specs/002-social-game-transform/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are REQUIRED. Every user story MUST include failing tests first, integrated validation at the relevant system boundary, regression coverage, and mypy for modified Python code.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this belongs to (e.g. `[US1]`, `[US2]`, `[US3]`)
- Include exact file paths in descriptions

## Path Conventions

- CLI pipeline code lives under `data_creation/`
- CLI tests live under `data_creation/tests/`
- Game contracts live under `games/`
- Game contract tests live under `tests/games/`
- Feature docs live under `specs/002-social-game-transform/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Refresh the feature fixtures and test scaffolds so the implementation work is driven by the current dual-game design rather than the stale beauty-contest-only behavior.

- [X] T001 Update the feature coverage notes in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/features/social_game_transform.feature` for `beauty_contest` plus `escalation_game`
- [X] T002 [P] Review the existing CLI test scaffold in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py` and mark the beauty-contest-only assumptions that must change
- [X] T003 [P] Review the current `EscalationGameScenario` contract in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/games/escalation_game.py` against the clarified `previous_actions` requirements

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared mapping and history-normalization primitives that every story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Extend the social-game mapping structure in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` to carry scenario class, canonical game name, artifact filenames, prompt wording, and deterministic injected fields
- [X] T005 [P] Add deterministic game-field injection helpers in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` for canonical `game_name`, payoff data, and provenance enrichment
- [X] T006 [P] Add `Escalation_Game` history normalization and validation helpers in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/games/escalation_game.py` so optional `previous_actions` can coexist with fallback `previous_actions_length`
- [X] T007 Update mapped scenario-class validation in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` so success rows validate through the explicit target class instead of hardcoded `BeautyContestScenario`
- [X] T008 [P] Add or update shared test fixtures for Beauty Contest rows and Escalation Game rows in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Produce Load-Ready Game Cases (Priority: P1) 🎯 MVP

**Goal**: Transform curated `beauty_contest` and plain `escalation_game` source rows into success-only datasets that load through their real scenario classes.

**Independent Test**: Run the CLI on a small fixture for each supported social game and verify that successful rows load through `BeautyContestScenario(**data)` or `EscalationGameScenario(**data)` while invalid rows land only in failure artifacts.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T009 [P] [US1] Add failing CLI integration coverage for dual-game success datasets in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py`
- [X] T010 [P] [US1] Add failing Escalation Game history-contract coverage for explicit `previous_actions`, fallback `previous_actions_length`, and mismatch rejection in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/tests/games/test_escalation_game_config.py`
- [X] T011 [P] [US1] Add failing game-load validation coverage for transformed Escalation Game rows in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/tests/games/test_escalation_game_config.py`

### Implementation for User Story 1

- [X] T012 [US1] Generalize prompt-pack loading and per-game prompt wording in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` for `beauty_contest` and `escalation_game`
- [X] T013 [US1] Implement per-game structural field injection and mapped artifact writing in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py`
- [X] T014 [US1] Extend `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/games/escalation_game.py` so `previous_actions` is optional but authoritative, with `previous_actions_length` fallback support
- [X] T015 [US1] Update row transformation and constructor validation flow in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` to support both scenario classes
- [X] T016 [US1] Run targeted pytest for `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py` and `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/tests/games/test_escalation_game_config.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Swap Prompt Assets By Social Game (Priority: P2)

**Goal**: Keep prompt assembly explicit per selected social game, including the temporary first-release rule that `escalation_game` may reuse the Beauty Contest few-shot asset while still enforcing its own runtime contract.

**Independent Test**: Verify that the CLI accepts the two supported games, rejects unsupported ones, assembles the prompt from the shared rubric plus the selected few-shot asset, and preserves correct runtime validation regardless of asset reuse.

### Tests for User Story 2 ⚠️

- [X] T017 [P] [US2] Add failing tests for explicit supported-game mapping and unsupported-game rejection in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py`
- [X] T018 [P] [US2] Add failing tests for prompt-pack assembly with `escalation_game` reusing the Beauty Contest few-shot asset in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py`

### Implementation for User Story 2

- [X] T019 [US2] Implement explicit dual-game prompt-asset mapping and unsupported-game rejection in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py`
- [X] T020 [US2] Keep the prompt-building flow simple in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` while allowing `escalation_game` to reuse the Beauty Contest few-shot asset by explicit choice
- [X] T021 [US2] Run targeted pytest for `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py` covering supported-game switching and prompt assembly

**Checkpoint**: At this point, User Stories 1 and 2 should both work independently

---

## Phase 5: User Story 3 - Resume And Audit A Long Transformation Run (Priority: P3)

**Goal**: Preserve deterministic resume behavior, artifact accounting, and run metadata across both supported social games.

**Independent Test**: Run the CLI twice on partial fixtures for each supported game and confirm that completed identities are not duplicated, skipped rows are recorded, and run metadata reconciles to the full processed row count.

### Tests for User Story 3 ⚠️

- [X] T022 [P] [US3] Add failing dual-game resume and rerun coverage in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py`
- [X] T023 [P] [US3] Add failing run-metadata and artifact-accounting coverage for both supported games in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py`

### Implementation for User Story 3

- [X] T024 [US3] Generalize artifact path selection and completed-identity bookkeeping in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` for both supported games
- [X] T025 [US3] Reconcile success, failure, skipped, and metadata outputs per supported game in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py`
- [X] T026 [US3] Run targeted pytest for `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py` covering resume, rerun, and artifact accounting

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup, regression validation, docs, and research-facing evidence.

- [X] T027 [P] Update usage documentation in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/README.md` for dual-game support and `Escalation_Game` history semantics
- [X] T028 [P] Update quick usage and evidence notes in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/quickstart.md` if implementation details shifted during TDD
- [X] T029 Refactor `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` and `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/games/escalation_game.py` for clarity after all story tests pass
- [X] T030 [P] Add any missing regression checks for dual-game artifact structure in `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/tests/test_transform_social_game_cases.py`
- [X] T031 Run `mypy` for `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/data_creation/transform_social_game_cases.py` and `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/games/escalation_game.py`
- [X] T032 Run quickstart validation from `/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/quickstart.md` and capture artifact evidence for both supported games

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion - this is the MVP
- **User Story 2 (Phase 4)**: Depends on Foundational completion and reuses the generalized mapping introduced for US1
- **User Story 3 (Phase 5)**: Depends on Foundational completion and the artifact flow preserved through US1
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - no dependency on later stories
- **User Story 2 (P2)**: Can start after Foundational, but is simplest after US1 establishes the dual-game mapping path
- **User Story 3 (P3)**: Can start after Foundational, but depends conceptually on the success/failure artifact model exercised by US1

### Within Each User Story

- Tests MUST be written and fail before implementation
- Shared mapping and validation helpers come before story-specific behavior
- Scenario contract changes come before final pipeline validation
- Story-level pytest verification comes before moving to the next story

### Parallel Opportunities

- T002 and T003 can run in parallel
- T005, T006, and T008 can run in parallel after T004 begins defining the mapping shape
- T009, T010, and T011 can run in parallel
- T017 and T018 can run in parallel
- T022 and T023 can run in parallel
- T027, T028, and T030 can run in parallel once implementation is complete

---

## Parallel Example: User Story 1

```bash
# Launch the failing tests for User Story 1 together:
Task: "T009 [US1] Add failing CLI integration coverage in data_creation/tests/test_transform_social_game_cases.py"
Task: "T010 [US1] Add failing Escalation Game history-contract coverage in tests/games/test_escalation_game_config.py"
Task: "T011 [US1] Add failing game-load validation coverage in tests/games/test_escalation_game_config.py"

# After the tests fail, implement the smallest passing changes:
Task: "T012 [US1] Generalize prompt-pack loading in data_creation/transform_social_game_cases.py"
Task: "T013 [US1] Implement per-game structural field injection in data_creation/transform_social_game_cases.py"
Task: "T014 [US1] Extend games/escalation_game.py for optional explicit previous_actions"
Task: "T015 [US1] Update row transformation and constructor validation flow in data_creation/transform_social_game_cases.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run the targeted CLI and scenario-contract tests for both supported games
5. Demo the CLI on small fixtures for `beauty_contest` and `escalation_game`

### Incremental Delivery

1. Build the explicit dual-game mapping and contract-validation foundation
2. Deliver dual-game loadable transformation as the MVP
3. Tighten prompt-asset switching and unsupported-game rejection
4. Reconfirm resume, rerun, and artifact bookkeeping across both games
5. Finish with docs, `mypy`, and quickstart validation

### Parallel Team Strategy

With multiple developers:

1. One developer stabilizes the mapping and artifact logic in `data_creation/transform_social_game_cases.py`
2. A second developer writes the failing CLI and prompt-asset tests in `data_creation/tests/test_transform_social_game_cases.py`
3. A third developer extends `games/escalation_game.py` and its contract tests in `tests/games/test_escalation_game_config.py`
4. After Foundation is merged, story work can proceed with low conflict because the main write scopes are the CLI module, the Escalation game contract, and the test modules

---

## Notes

- `[P]` tasks touch different files or independent test slices
- Every task includes an exact file path
- The MVP scope is User Story 1
- Success datasets must remain success-only throughout implementation
- Avoid adding a plugin system, auto-discovery, or generic service layer for this feature
