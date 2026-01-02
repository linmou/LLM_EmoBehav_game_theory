# Data Model: Prisoner's Dilemma Emotion Steering Similarity

**Feature**: 002-pd-steering-sim  
**Last Updated**: 2025-12-08

## Entities

### SteeringRunConfig

Represents a single config-driven analysis run over the PD game-theory benchmark.

- `id`: Logical run identifier or label.  
- `model_name`: Name used in configs and logs (e.g., `Qwen2.5-1.5B-Instruct`).  
- `model_path`: Filesystem path to the model weights.  
- `benchmark_name`: Benchmark identifier (e.g., `game_theory`).  
- `pd_task_id`: Specific PD task identifier within the game-theory benchmark.  
- `emotions`: List of emotion labels to steer (e.g., `["anger", "fear"]`).  
- `intensities`: List of steering intensities to test (floats).  
- `base_config_path`: Path to the YAML config driving the run.  
- `pd_defection_vector_dir`: Directory containing layer-wise PD defection vectors.  
- `output_dir`: Directory for similarity outputs and summaries.

### PDSample

Represents a single PD scenario and the model’s choice before/after steering.

- `sample_id`: Unique identifier from `raw_results.json`.  
- `scenario_id`: Optional identifier for the underlying PD scenario.  
- `baseline_choice`: Model’s choice without steering (e.g., `"cooperate"` or `"defect"`).  
- `steered_choice`: Model’s choice with steering applied.  
- `switched_to_defect`: Boolean flag derived from choices.  
- `metadata`: Additional fields from the original benchmark (round index, payoff info, etc.).

### LayerVector

Represents a single layer’s direction or activation.

- `layer_index`: Integer index of the transformer block.  
- `vector`: 1D float array representing either a PD defection direction or a hidden state at the last input token.

### SteeringCondition

Represents a single emotion/intensity condition within a run.

- `id`: Unique identifier (e.g., `"anger_1.5"`).  
- `emotion`: Emotion label (e.g., `"anger"`).  
- `intensity`: Steering intensity (float).  
- `run_config_id`: Reference to `SteeringRunConfig.id`.

### LayerSimilarityRecord

Represents similarity measurements for one sample, one steering condition, and one layer.

- `sample_id`: Reference to `PDSample.sample_id`.  
- `steering_condition_id`: Reference to `SteeringCondition.id`.  
- `layer_index`: Integer layer index.  
- `similarity_baseline`: Cosine similarity between baseline hidden state and PD defection vector.  
- `similarity_steered`: Cosine similarity between steered hidden state and PD defection vector.  
- `similarity_delta`: Difference (`similarity_steered - similarity_baseline`).

### GroupSummary

Aggregated statistics for switchers vs non-switchers per layer and steering condition.

- `steering_condition_id`: Reference to `SteeringCondition.id`.  
- `layer_index`: Integer layer index.  
- `group_label`: `"switcher"` or `"non-switcher"`.  
- `mean_similarity_delta`: Mean of `similarity_delta` for this group.  
- `std_similarity_delta`: Standard deviation for this group.  
- `n_samples`: Number of samples in the group.

## Relationships

- One `SteeringRunConfig` has many `SteeringCondition` records.  
- Each `PDSample` can have many `LayerSimilarityRecord` entries (one per steering condition and layer).  
- Each `LayerSimilarityRecord` belongs to exactly one `SteeringCondition`.  
- Each `GroupSummary` aggregates many `LayerSimilarityRecord` entries sharing the same `(steering_condition_id, layer_index, group_label)`.

## Constraints

- The tuple `(sample_id, steering_condition_id, layer_index)` must be unique in `LayerSimilarityRecord`.  
- `switched_to_defect` is `True` iff `baseline_choice` is non-defect and `steered_choice` equals `"defect"` (after any normalization).  
- Similarity computations only include layers for which a PD defection vector is available; missing layers are skipped and reported.  
- Outputs for each `SteeringRunConfig` are written under a single `output_dir` tree to keep runs isolated and reproducible.

