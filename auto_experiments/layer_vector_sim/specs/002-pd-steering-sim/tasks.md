# Tasks: Prisoner's Dilemma Emotion Steering Similarity

**Input**: Design documents from `/specs/002-pd-steering-sim/`  
**Prerequisites**: `plan.md` (required), `spec.md` (required), `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: This feature uses a TDD workflow. Every functional slice includes explicit test tasks that should be implemented and run before production code.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure repository structure, environment, and base test scaffolding exist for PD steering similarity work.

- [ ] T001 Create analysis module skeleton `auto_experiments/layer_vector_sim/pd_steering_similarity/__init__.py`
- [ ] T002 Create main entrypoint module `auto_experiments/layer_vector_sim/pd_steering_similarity/run_pd_steering_similarity.py`
- [ ] T003 Create config schema module `auto_experiments/layer_vector_sim/pd_steering_similarity/config_schema.py`
- [ ] T004 Create test package for PD similarity `tests/auto_experiments/test_pd_steering_similarity/__init__.py`
- [ ] T005 [P] Add base unit test file `tests/auto_experiments/test_pd_steering_similarity/test_config_parsing.py`
- [ ] T006 [P] Add base unit test file `tests/auto_experiments/test_pd_steering_similarity/test_similarity_math.py`
- [ ] T007 [P] Add integration test skeleton `tests/auto_experiments/test_pd_steering_similarity/test_full_pipeline_smoke.py`

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Provide shared utilities and loaders required by all user stories.

- [ ] T008 Implement YAML config loading and validation in `auto_experiments/layer_vector_sim/pd_steering_similarity/config_schema.py`
- [ ] T009 Implement config parsing tests in `tests/auto_experiments/test_pd_steering_similarity/test_config_parsing.py`
- [ ] T010 Implement steering vector loader wrapper that reuses `EmotionExperiment` logic in `auto_experiments/layer_vector_sim/pd_steering_similarity/steering_loader.py`
- [ ] T011 Add unit tests for steering vector loader in `tests/auto_experiments/test_pd_steering_similarity/test_steering_loader.py`
- [ ] T012 Implement PD defection vector loader for layer vectors in `auto_experiments/layer_vector_sim/pd_steering_similarity/pd_defection_loader.py`
- [ ] T013 Add unit tests for PD defection vector loader in `tests/auto_experiments/test_pd_steering_similarity/test_pd_defection_loader.py`
- [ ] T014 Implement cosine similarity and similarity-delta helpers in `auto_experiments/layer_vector_sim/pd_steering_similarity/similarity_utils.py`
- [ ] T015 Add unit tests for similarity utilities in `tests/auto_experiments/test_pd_steering_similarity/test_similarity_math.py`
- [ ] T016 Implement run-or-load helper for existing PD benchmark raw results in `auto_experiments/layer_vector_sim/pd_steering_similarity/benchmark_io.py`
- [ ] T017 Add tests for benchmark IO helper in `tests/auto_experiments/test_pd_steering_similarity/test_benchmark_io.py`

## Phase 3: User Story 1 - Analyze PD switchers after emotion steering (Priority: P1)

**Goal**: For each switcher sample, compute per-layer similarity between hidden states (baseline vs steered) and PD defection vectors.

**Independent Test**: Given a small PD fixture dataset and toy steering/defection vectors, the pipeline produces per-layer similarity metrics for switcher samples only and writes them to disk.

### Tests for User Story 1

- [ ] T018 [P] [US1] Add fixture PD raw_results and steering config under `tests/auto_experiments/test_pd_steering_similarity/fixtures/`
- [ ] T019 [P] [US1] Implement unit tests for identifying switcher vs non-switcher samples in `tests/auto_experiments/test_pd_steering_similarity/test_sample_grouping.py`
- [ ] T020 [P] [US1] Implement unit tests for per-layer similarity computation given hidden states and PD defection vectors in `tests/auto_experiments/test_pd_steering_similarity/test_similarity_per_layer.py`
- [ ] T021 [US1] Implement integration test that runs the pipeline on fixtures and checks switcher similarity output in `tests/auto_experiments/test_pd_steering_similarity/test_full_pipeline_smoke.py`

### Implementation for User Story 1

- [ ] T022 [P] [US1] Implement `PDSample` and grouping logic (switcher vs non-switcher) in `auto_experiments/layer_vector_sim/pd_steering_similarity/sample_grouping.py`
- [ ] T023 [P] [US1] Implement hidden state extraction interface that hooks into PD benchmark runs in `auto_experiments/layer_vector_sim/pd_steering_similarity/hidden_state_capture.py`
- [ ] T024 [P] [US1] Implement per-layer similarity computation core using `similarity_utils` in `auto_experiments/layer_vector_sim/pd_steering_similarity/layer_similarity.py`
- [ ] T025 [US1] Implement writer for per-sample, per-layer similarity records in `auto_experiments/layer_vector_sim/pd_steering_similarity/output_writer.py`
- [ ] T026 [US1] Wire config, loaders, hidden-state capture, and similarity computation together in `auto_experiments/layer_vector_sim/pd_steering_similarity/run_pd_steering_similarity.py` for the US1 path (switcher similarity only)

**Checkpoint**: Running the analysis on fixtures produces correct similarity records for switcher samples and passes all US1 tests.

## Phase 4: User Story 2 - Compare switchers vs stable samples (Priority: P2)

**Goal**: Compare similarity shifts between switchers and non-switchers per layer.

**Independent Test**: Given fixtures with both switchers and non-switchers, the analysis produces group-level statistics and highlights differences between groups per layer.

### Tests for User Story 2

- [ ] T027 [P] [US2] Extend fixtures to include non-switcher samples in `tests/auto_experiments/test_pd_steering_similarity/fixtures/`
- [ ] T028 [P] [US2] Implement unit tests for computing group summaries (mean, std, n) per layer in `tests/auto_experiments/test_pd_steering_similarity/test_group_summaries.py`
- [ ] T029 [US2] Implement integration test that checks presence and correctness of switcher vs non-switcher summaries in `tests/auto_experiments/test_pd_steering_similarity/test_full_pipeline_group_comparison.py`

### Implementation for User Story 2

- [ ] T030 [P] [US2] Implement aggregation of `LayerSimilarityRecord` into `GroupSummary` structures in `auto_experiments/layer_vector_sim/pd_steering_similarity/group_aggregation.py`
- [ ] T031 [P] [US2] Implement serialization of group summaries to CSV/JSON in `auto_experiments/layer_vector_sim/pd_steering_similarity/output_writer.py`
- [ ] T032 [US2] Integrate group summary generation into `run_pd_steering_similarity.py` controlled by config flags

**Checkpoint**: Outputs now include per-layer group-level similarity deltas for switchers and non-switchers and all US2 tests pass.

## Phase 5: User Story 3 - Rank emotions by PD defection similarity shift (Priority: P3)

**Goal**: Aggregate similarity shifts by emotion and intensity and rank steering conditions by their effect on PD defection alignment.

**Independent Test**: Given fixtures with multiple emotions and intensities, the analysis outputs a ranking of steering conditions by average similarity shift toward PD defection vectors.

### Tests for User Story 3

- [ ] T033 [P] [US3] Extend fixtures to include multiple emotions and intensities in `tests/auto_experiments/test_pd_steering_similarity/fixtures/`
- [ ] T034 [P] [US3] Implement unit tests for computing emotion-level rankings from `GroupSummary` data in `tests/auto_experiments/test_pd_steering_similarity/test_emotion_rankings.py`
- [ ] T035 [US3] Implement integration test that validates the produced ranking file and top emotion/intensity identification in `tests/auto_experiments/test_pd_steering_similarity/test_full_pipeline_emotion_ranking.py`

### Implementation for User Story 3

- [ ] T036 [P] [US3] Implement aggregation logic that computes average similarity deltas per emotion and intensity in `auto_experiments/layer_vector_sim/pd_steering_similarity/emotion_aggregation.py`
- [ ] T037 [P] [US3] Implement writer for emotion-level ranking artifacts (e.g., CSV/JSON) in `auto_experiments/layer_vector_sim/pd_steering_similarity/output_writer.py`
- [ ] T038 [US3] Hook emotion-level aggregation and ranking into `run_pd_steering_similarity.py` and ensure config toggles are respected

**Checkpoint**: Emotion-level summary outputs rank all configured emotions by similarity shift and identify top emotion/intensity combinations; all US3 tests pass.

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements across all user stories.

- [ ] T039 [P] Add CLI argument parsing help and usage examples to `auto_experiments/layer_vector_sim/pd_steering_similarity/run_pd_steering_similarity.py`
- [ ] T040 [P] Document PD steering similarity usage and config examples in `specs/002-pd-steering-sim/quickstart.md`
- [ ] T041 Add docstrings and inline comments to key analysis modules under `auto_experiments/layer_vector_sim/pd_steering_similarity/`
- [ ] T042 [P] Add additional edge-case tests (missing layers, missing vectors, malformed samples) in `tests/auto_experiments/test_pd_steering_similarity/test_edge_cases.py`
- [ ] T043 Run mypy on new modules and fix typing issues in `auto_experiments/layer_vector_sim/pd_steering_similarity/`
- [ ] T044 Ensure all tests pass (unit and integration) and record any known limitations in `specs/002-pd-steering-sim/research.md`

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies – can start immediately.  
- Foundational (Phase 2): Depends on Phase 1 – blocks all user stories.  
- User Story phases (3–5): Depend on completion of Phase 2; can run in parallel across different files once foundational utilities are stable.  
- Polish (Phase 6): Depends on completion of desired user stories.

### User Story Dependencies

- User Story 1 (P1): Depends on foundational config, loaders, and similarity utilities. No dependency on US2 or US3.  
- User Story 2 (P2): Depends on US1 similarity record generation (uses same records but adds aggregation).  
- User Story 3 (P3): Depends on US2 group summaries (emotion-level rankings build on aggregated data).

### Parallel Opportunities

- Tasks marked `[P]` operate on distinct files and can be implemented in parallel once their prerequisites are in place.  
- Within each user story, independent unit tests and aggregation utilities can be developed in parallel after foundational modules are available.  
- User Stories 2 and 3 can overlap partially once US1’s similarity record generation is stable and tested.

## Implementation Strategy

- MVP: Complete Phases 1–3 (Setup, Foundational, User Story 1), then run the analysis end-to-end for switcher similarity only.  
- Incremental: Add User Story 2 (group comparison) then User Story 3 (emotion-level rankings), validating after each story using the associated tests and fixtures.  
- Always follow TDD: write or extend the relevant test(s) for each task before implementing the corresponding production code.

