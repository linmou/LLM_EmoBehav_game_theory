# Research Notes: Prisoner's Dilemma Emotion Steering Similarity

**Feature**: 002-pd-steering-sim  
**Last Updated**: 2025-12-08

## Decision 1: Language and Runtime

- Decision: Python 3.10+ in the existing `llm_fresh` conda environment.  
- Rationale: The repository and experiment scripts are already Python-based and documented to run inside `llm_fresh`, so no new runtime is required.  
- Alternatives considered: Introducing a separate microservice or different language would add deployment and integration overhead with no clear benefit for an offline analysis pipeline.

## Decision 2: Hidden-state Position for Similarity

- Decision: Use the hidden state at the last input token (just before the first generated token) for all layers.  
- Rationale: This position captures the model’s internal state at the decision boundary implied by the PD prompt, independent of generation length, and matches the clarified feature spec. It is straightforward to extract from vLLM hooks.  
- Alternatives considered:
  - Last generated token (would mix decision representation with downstream generation artifacts).  
  - Averaging over all tokens (more expensive and harder to interpret layer-wise effects).

## Decision 3: Similarity Metric

- Decision: Cosine similarity between each per-layer hidden state vector and its corresponding PD defection direction vector.  
- Rationale: PD defection vectors encode directions in representation space; cosine similarity is scale-invariant and standard for such alignment measurements, and is consistent with how PD defection directions were originally derived.  
- Alternatives considered: Dot product (sensitive to vector norms) and Euclidean distance (less interpretable for directional alignment).

## Decision 4: Steering Vector Loading

- Decision: Reuse `EmotionExperiment` (from `emotion_experiment_engine.experiment`) logic to locate and load emotion steering vectors per layer.  
- Rationale: Keeps a single source of truth for steering-vector formats, paths, and any future changes; avoids duplicating loading conventions in a separate analysis-only module.  
- Alternatives considered: Custom loaders in the analysis code (rejected as duplication and potential source of drift).

## Decision 5: Analysis Placement and Inputs

- Decision: Implement the PD steering similarity analysis as a Python module/CLI under `auto_experiments/layer_vector_sim` that consumes:
  - PD game-theory raw results JSON (for switcher vs non-switcher labels),  
  - PD defection layer vectors from `auto_experiments/task_similarity/results/steering_vectors/.../layer_vectors`,  
  - A YAML config describing model, benchmark, steering emotions, and intensities.  
- Rationale: Co-locates all PD layer-vector analysis with existing `layer_vector_sim` experiments; leverages existing directory conventions; keeps the feature as a focused offline analysis tool instead of a new service.  
- Alternatives considered: Implementing under `emotion_experiment_engine` (would mix experiment orchestration with offline analysis) or creating a new top-level package (unnecessary fragmentation).

## Decision 6: Group Comparison Logic

- Decision: Define two main sample groups using `raw_results.json`:
  - **Switchers**: samples where the baseline choice is "cooperate" (or equivalent) and the steered choice is "defect".  
  - **Non-switchers**: all other samples with a defined baseline choice.  
- Rationale: This directly targets the research question: whether emotion steering shifts representations toward PD defection directions specifically for samples whose behavior flips to defection. Non-switchers provide a natural comparison group.  
- Alternatives considered: Finer-grained grouping (e.g., defect→cooperate, cooperate→cooperate) was deemed unnecessary for the initial PD-defection-focused analysis and can be added later if needed.

