---

description: "Task list for Shuffle Behavior Option Order and Dual Choice Ratios"
---

# Tasks: Shuffle Behavior Option Order and Dual Choice Ratios

**Input**: Design documents from `/specs/001-shuffle-choice-order/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: This feature will be implemented with TDD. Every behavior change must have a failing pytest first, then minimal implementation, then refactor.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create feature directory structure (already present) and confirm design docs in specs/001-shuffle-choice-order/
- [x] T002 [P] Verify conda environment and minimal pytest run in `${USER_HOME}/LLM_EmoBehav_game_theory_autoexp_worktree` using `conda activate llm_fresh` and `pytest -q`
- [x] T003 [P] Skim emotion_experiment_engine/datasets/games.py to confirm current GameTheoryDataset behavior and choice_ratio metrics
- [x] T004 [P] Skim games/game.py and games/* to understand BehaviorChoices and find_behavior_from_decision contracts
- [x] T005 [P] Skim emotion_experiment_engine/experiment.py around _save_results and split-metrics persistence

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared understanding and minimal utilities needed by all user stories.

- [x] T006 Add any missing imports and type aliases needed for shuffling and behavior-label mapping in emotion_experiment_engine/datasets/games.py
- [x] T007 [P] Confirm existing tests for GameTheoryDataset pass: run `pytest emotion_experiment_engine/tests/unit/test_game_evaluation_extraction.py emotion_experiment_engine/tests/unit/test_split_choice_ratio.py emotion_experiment_engine/tests/unit/test_game_stats_analysis.py -q`
- [ ] T008 [P] Add small internal helper (if needed) in emotion_experiment_engine/datasets/games.py to look up behavior label by option_id from BenchmarkItem.metadata["options"] while keeping code simple

## Phase 3: User Story 1 - Separate behavior preference from position bias (Priority: P1)

**Goal**: Randomize per-sample behavior option order while keeping index-based metrics intact, and ensure index→behavior mapping is recoverable.

**Independent Test**: Run a small game_theory experiment twice with different seeds and verify that some samples have swapped behavior labels across option indices while `summary_choice_ratio.csv` still reflects correct id-based ratios and per-decision metadata preserves index→behavior mapping.

### Tests for User Story 1

### Tests for User Story 1

- [x] T009 [P] [US1] Add pytest module tests/emotion_experiment_engine/datasets/test_games_dataset_shuffle.py with a fixture that builds a tiny synthetic raw scenario list and checks that GameTheoryDataset shuffles option order across items
- [x] T010 [P] [US1] In tests/emotion_experiment_engine/datasets/test_games_dataset_shuffle.py, add a test asserting that each BenchmarkItem.metadata["options"] entry contains {"id", "text", "behavior"} and that ids are 1-based contiguous after shuffling
- [x] T011 [P] [US1] In tests/emotion_experiment_engine/datasets/test_games_dataset_shuffle.py, add a test that runs GameTheoryDataset twice with different random seeds and verifies at least one sample’s behavior (the `behavior` field in options metadata) appears at different option_id positions across runs

### Implementation for User Story 1

### Implementation for User Story 1

- [x] T012 [US1] In emotion_experiment_engine/datasets/games.py, update _load_and_parse_data to construct options from scenario.get_behavior_choices().get_choices() and attach a behavior label via scenario.find_behavior_from_decision for each choice
- [x] T013 [US1] In _load_and_parse_data, introduce a simple per-scenario shuffle of the options list (e.g., using random.shuffle) and then reassign option ids to 1..N in shuffled order before storing in BenchmarkItem.metadata["options"]
- [x] T014 [US1] Ensure that default _build_items_from_raw continues to work for non-game_theory tasks and does not attempt to attach behavior labels when scenarios are not available
- [x] T015 [US1] Wire a deterministic random seed hook (e.g., via numpy.random or Python’s random seeded from existing experiment config) so that shuffling can be controlled externally for reproducibility in tests
- [x] T016 [US1] Verify via pytest that existing choice_ratio tests (emotion_experiment_engine/tests/unit/test_split_choice_ratio.py and emotion_experiment_engine/tests/unit/test_experiment_choice_ratio_persistence.py) still pass, confirming that shuffling does not break id-based aggregation

## Phase 4: User Story 2 - Report both behavior-level and index-level choice ratios (Priority: P2)

**Goal**: Extend GameTheoryDataset.compute_split_metrics so that it returns both id-based and behavior-based choice ratios, and persist behavior-level summaries alongside existing CSVs.

**Independent Test**: Given a synthetic set of ResultRecord instances with known chosen option indices and behavior labels, metrics from compute_split_metrics and the saved summary CSVs must match expected behavior-level and index-level ratios exactly (up to rounding).

### Tests for User Story 2

- [x] T017 [P] [US2] Add pytest module tests/emotion_experiment_engine/datasets/test_games_behavior_choice_ratio.py to verify that compute_split_metrics returns behavior-level ratios when BenchmarkItem.metadata["options"][i]["behavior"] is populated
- [x] T018 [P] [US2] In tests/emotion_experiment_engine/datasets/test_games_behavior_choice_ratio.py, create a small list of ResultRecord objects with metadata including options and scores, then assert that behavior-level counts and ratios are correct for multiple behaviors and emotions
- [x] T019 [P] [US2] Add pytest to emotion_experiment_engine/tests/unit/test_experiment_choice_ratio_persistence.py (or a new test) to assert that EmotionExperiment._save_results persists a new summary_behavior_ratio.csv with expected behavior-level ratios

### Implementation for User Story 2

- [x] T020 [US2] In emotion_experiment_engine/datasets/games.py, extend compute_split_metrics to read per-item options and map each ResultRecord.score (option_id) to a behavior label (from the `behavior` field in metadata), aggregating counts per (emotion, intensity[, repeat_id], behavior label)
- [x] T021 [US2] In compute_split_metrics, keep existing choice_ratio id-based payload unchanged and add a new behavior_choice_ratio structure with "overall" and "by_repeat" behavior-level rows
- [x] T022 [US2] In emotion_experiment_engine/experiment.py, extend _save_results to optionally write a summary_behavior_ratio.csv and, if needed, summary_behavior_ratio_by_repeat.csv when behavior_choice_ratio is present in split_metrics
- [x] T023 [US2] Confirm that summary_behavior_ratio.csv includes columns [emotion, intensity, behavior, ratio] and that sums of ratios per (emotion, intensity) group are approximately 1.0
- [x] T024 [US2] Run targeted pytest on the new tests and existing choice_ratio tests to ensure no regressions in id-based behavior

## Phase 4b: User Story 2 – Invalid or Missing Behavior Categories (FR-007)

**Goal**: Ensure scenarios with missing or ambiguous behavior categories are handled explicitly (skipped or rejected) and never silently pollute behavior-level ratios.

**Independent Test**: When a sample lacks a valid behavior category mapping for a chosen option, the behavior-level aggregation either skips that decision consistently or raises a clear error; this behavior is fully covered by tests.

### Tests for FR-007

- [x] T024a [P] [US2] In tests/emotion_experiment_engine/datasets/test_games_behavior_choice_ratio.py, add a test case where BenchmarkItem.metadata["options"] is missing a behavior entry for a chosen option_id and assert that compute_split_metrics fails fast with a clear exception message

### Implementation for FR-007

- [x] T024b [US2] In emotion_experiment_engine/datasets/games.py, update compute_split_metrics to validate that every chosen option_id encountered has a corresponding non-empty behavior category in metadata; if not, raise a ValueError with a concise explanation so the experiment can surface the data issue early

- [x] T033 [P] [US2] Extend tests in emotion_experiment_engine/tests/unit/test_games_behavior_choice_ratio.py to cover unmappable option_id values (e.g., 3 when only ids 1 and 2 exist), asserting they appear as an explicit \"unknown\" behavior bucket while still contributing to id-based choice_ratio
- [x] T034 [US2] In emotion_experiment_engine/datasets/games.py, update _behavior_choice_ratios so that when no option with id == chosen option_id is found, the decision is counted under a canonical \"unknown\" behavior label instead of raising, while still raising for matched options with missing/empty behavior
- [x] T035 [P] [US2] Run targeted pytest subset (test_split_choice_ratio.py, test_game_stats_analysis.py, test_games_behavior_choice_ratio.py, test_experiment_choice_ratio_persistence.py) to validate the new \"unknown\" behavior bucket semantics and confirm id-level ratios remain consistent (FR-006)

## Phase 5: User Story 3 - Inspect individual samples for audit and debugging (Priority: P3)

**Goal**: Make it easy to inspect individual decisions, including shuffled option order, chosen index, and behavior label, using existing JSON/CSV outputs.

**Independent Test**: A researcher can load raw_results.json and summary CSVs for a small run and manually trace at least one decision from the aggregated behavior/id ratios back to its underlying sample options, chosen index, and behavior label without ambiguity.

### Tests for User Story 3

- [x] T025 [P] [US3] Add a helper-based test in tests/emotion_experiment_engine/datasets/test_games_dataset_shuffle.py (or a new test) that round-trips a BenchmarkItem metadata options list and a simulated choice to a DecisionRecord-like structure and back, verifying that id and behavior are consistent

### Implementation for User Story 3

- [x] T026 [US3] Ensure that EmotionExperiment._save_results keeps raw_results.json unchanged except for enriched metadata, so that per-sample options and any future behavior labels in ResultRecord.metadata remain available for manual inspection
- [x] T027 [US3] Optionally add a tiny utility function or documented pattern (e.g., in result_analysis/README.md) showing how to load raw_results.json and map from aggregated ratios back to an example decision without introducing new dependencies

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Clean up, verify documentation, and ensure everything is easy to run.

- [x] T028 [P] Run targeted pytest subset for game_theory features: emotion_experiment_engine/tests/unit/test_split_choice_ratio.py, test_game_evaluation_extraction.py, test_game_stats_analysis.py, new tests for shuffling and behavior ratios
- [ ] T029 [P] Update tasks/game_theory_to_emotion_experiment_migration.md if needed to mention behavior-level ratios and shuffling semantics succinctly
- [ ] T030 Ensure quickstart.md in specs/001-shuffle-choice-order/ still matches the implemented behavior and test paths
- [ ] T031 [P] Do a light pass over emotion_experiment_engine/datasets/games.py to remove dead code and keep functions small and focused (no new abstractions unless absolutely necessary)
- [ ] T032 Run a small real experiment with game_theory benchmark (e.g., Prisoners_Dilemma) and visually inspect summary_choice_ratio.csv and summary_behavior_ratio.csv to confirm outputs match expectations
