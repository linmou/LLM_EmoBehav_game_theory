# Feature Specification: Prisoner's Dilemma Emotion Steering Similarity

**Feature Branch**: `002-pd-steering-sim`  
**Created**: 2025-12-08  
**Status**: Draft  
**Input**: User description: "for the game theory benchmark (see emotion_experiment_engine/benchmark_component_registry.py ), task: prisoners dilemma, i want to see after layer level emotion steering (+ steering vector (use the steering vectors loaded in emotion_experiment_engine.experiment.EmotionExperiment, use the same method to load them) ), if the hidden state show more similarity to the corresponding layers prisoners dilemma defection direction (find prisoners dilemma defection direction at auto_experiments/task_similarity/results/steering_vectors/Qwen2.5-1.5B-Instruct/20251201_112845/seed_20/layer_vectors) . especially for the data samples which switch from option cooperate to option defect (you can find them at results/new_game_theory/Qwen2.5-1.5B-Instruct_game_theory_Prisoners_Dilemma_20250930_214407/raw_results.json ), i want to see if their layer-level similarity is higher than the samples without change. And please also compare the emotion-level similarity, see if some emotions steering make layer hidden states more similar to PD defection direction. Use qwen2.5-1.5B-Instruction model. but keep everything configurable (currently, model, intensity, benchmark)as a config yaml"

## Clarifications

### Session 2025-12-08

- Q: Which token position should be used when computing similarity to PD defection vectors? → A: Use the hidden state at the last input token (just before the first generated token).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Analyze PD switchers after emotion steering (Priority: P1)

Researcher runs the PD benchmark with layer-level emotion steering and inspects whether hidden states for samples that switch from cooperate to defect move closer to the PD defection direction.

**Why this priority**: Core question is whether steering nudges switcher samples toward defection representations; without this, the study cannot validate the hypothesis.

**Independent Test**: Run the configured pipeline on the provided PD results file, focusing on switcher samples only, and generate a per-layer similarity report without needing other features.

**Acceptance Scenarios**:

1. **Given** a YAML config specifying model, benchmark, steering vectors, and switcher sample IDs, **When** the analyst runs the PD analysis, **Then** the system outputs per-layer similarity metrics between steered hidden states and PD defection vectors for each switcher sample.  
2. **Given** the same config, **When** the analyst requests summary statistics, **Then** the system returns aggregated similarity shifts (baseline vs steered) for switchers across layers.

---

### User Story 2 - Compare switchers vs stable samples (Priority: P2)

Researcher compares similarity changes for switch-to-defect samples against samples whose choice did not change.

**Why this priority**: Establishes whether steering effects are specific to behavior changes or general across all samples.

**Independent Test**: Run the analysis with group splitting (switchers vs non-switchers) and verify the report highlights group-level similarity differences.

**Acceptance Scenarios**:

1. **Given** grouped PD samples (switchers and non-switchers), **When** the analysis runs, **Then** the report contains per-layer group means and deltas comparing the two groups.

---

### User Story 3 - Rank emotions by PD defection similarity shift (Priority: P3)

Researcher wants to see which emotions and intensities increase alignment with PD defection vectors.

**Why this priority**: Helps select the most effective steering emotion/intensity for PD defection behavior.

**Independent Test**: Execute the analysis across configured emotions/intensities and confirm the output ranks emotions by similarity shift.

**Acceptance Scenarios**:

1. **Given** multiple emotions and intensities in the YAML config, **When** the analysis completes, **Then** the output ranks emotions by how much they increase similarity to PD defection vectors and highlights the top-performing settings.

---

### Edge Cases

- Missing steering vector or PD defection vector for a layer is detected and the layer is flagged or excluded with a reason in the report.
- Samples lacking decision metadata (cannot determine switch/no-switch) are skipped with a traceable note.
- Configured emotion or intensity not available for the model is reported and the run aborts or skips that setting without corrupting other results.
- Zero or negative steering intensity defaults to a neutral run and is labeled accordingly in outputs.
- Layer indexing mismatches between model hidden states and stored PD defection vectors are detected and resolved or surfaced before calculations proceed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load emotion steering vectors per layer using the same source and method defined in `emotion_experiment_engine.experiment.EmotionExperiment`, driven entirely by YAML config (model, emotion, intensity).
- **FR-002**: System MUST load PD defection direction vectors per layer from the provided directory and align them to the model's layer numbering.
- **FR-003**: System MUST extract hidden states for PD benchmark runs with and without steering for the configured model and benchmark task.
- **FR-004**: System MUST compute per-layer similarity (e.g., cosine) between hidden states at the last input token position (just before the first generated token) and PD defection vectors for both baseline and steered runs to derive a similarity shift per sample.
- **FR-005**: System MUST identify "switch to defect" samples from `raw_results.json` (cooperate → defect) and label remaining samples as non-switchers.
- **FR-006**: System MUST compare similarity distributions between switchers and non-switchers per layer and surface group-level deltas in the report.
- **FR-007**: System MUST aggregate similarity shifts by emotion and intensity, rank emotions by their impact on alignment with PD defection vectors, and expose these rankings in outputs.
- **FR-008**: System MUST output human-readable artifacts (e.g., tables or charts) summarizing per-layer and per-emotion findings, along with machine-readable data for downstream analysis.
- **FR-009**: System MUST allow model choice, steering intensity, emotions to test, benchmark selection, input paths, and output destinations to be set via a YAML config without code changes.

### Key Entities *(include if feature involves data)*

- **Steering Condition**: Combination of emotion, intensity, and model settings used to produce steered hidden states.
- **PD Defection Vector**: Layer-indexed direction representing PD defection behavior used as the similarity target.
- **Sample Outcome Group**: Classification of PD benchmark samples into switchers (cooperate → defect) and non-switchers based on recorded choices.
- **Layer Similarity Record**: Per-sample, per-layer similarity values for baseline and steered runs, including shifts and group labels.

### Assumptions

- Similarity is measured with cosine distance/angle consistent with how PD defection vectors were derived.  
- Hidden states are taken at the last input token position (just before the first generated token) to align with the PD defection vector reference point.  
- Default model is Qwen2.5-1.5B-Instruct, but configs may override model path/name and intensity values.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single configured run produces per-layer similarity metrics for at least 95% of analyzed samples and all layers that have both steering and PD defection vectors.
- **SC-002**: Reports include group-level (switcher vs non-switcher) similarity deltas for every layer within one execution, without manual data editing.
- **SC-003**: Emotion-level summaries rank all configured emotions by similarity shift and clearly identify the top emotion/intensity combination in the final outputs.
- **SC-004**: Changing model, intensity, or benchmark solely via the YAML config and rerunning completes successfully and regenerates all outputs in that new configuration.
