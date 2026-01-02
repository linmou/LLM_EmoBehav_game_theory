# Quickstart: Prisoner's Dilemma Emotion Steering Similarity

**Feature**: 002-pd-steering-sim  
**Last Updated**: 2025-12-08

This guide explains how to run the PD emotion steering similarity analysis once the implementation is in place.

## Prerequisites

- Conda environment `llm_fresh` available and activated:

```bash
conda activate llm_fresh
```

- Model weights for Qwen2.5-1.5B-Instruct available on disk.  
- Existing PD game-theory raw results JSON:
  - `results/new_game_theory/Qwen2.5-1.5B-Instruct_game_theory_Prisoners_Dilemma_*/raw_results.json`
- Existing PD defection layer vectors:
  - `auto_experiments/task_similarity/results/steering_vectors/Qwen2.5-1.5B-Instruct/.../layer_vectors/`

## 1. Create a YAML Config

Create a config file (for example):

```yaml
# pd_steering_similarity_qwen2.5-1.5B.yaml
model:
  name: Qwen2.5-1.5B-Instruct
  path: /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct

benchmark:
  name: game_theory
  task: Prisoners_Dilemma
  raw_results_path: results/new_game_theory/Qwen2.5-1.5B-Instruct_game_theory_Prisoners_Dilemma_20250930_214407/raw_results.json

steering:
  emotions: ["anger", "fear", "sadness", "happiness"]
  intensities: [0.5, 1.0, 1.5]
  loader: emotion_experiment_engine.experiment.EmotionExperiment

pd_defection_vectors:
  dir: auto_experiments/task_similarity/results/steering_vectors/Qwen2.5-1.5B-Instruct/20251201_112845/seed_20/layer_vectors

output:
  dir: results/pd_steering_similarity/Qwen2.5-1.5B-Instruct/
```

Adjust paths as needed for your environment.

## 2. Run the Analysis (Planned Interface)

After implementation, the planned entrypoint will be a Python module under `auto_experiments/layer_vector_sim`, for example:

```bash
python -m auto_experiments.layer_vector_sim.run_pd_steering_similarity \
  --config pd_steering_similarity_qwen2.5-1.5B.yaml
```

This command will:

- Load the PD game-theory samples and identify switchers vs non-switchers.  
- Load PD defection vectors and emotion steering vectors.  
- Extract hidden states at the last input token for each layer (baseline and steered).  
- Compute cosine similarity to PD defection directions per layer.  
- Write per-sample and aggregated summaries under `output.dir`.

## 3. Inspect Results

Typical outputs (to be finalized during implementation) will include:

- Per-layer CSV/JSON with similarity_baseline, similarity_steered, similarity_delta.  
- Group summaries comparing switchers vs non-switchers per layer and steering condition.  
- Emotion-level summaries ranking emotions by average similarity shift toward PD defection.

Use these files for downstream plotting and statistical analysis.

