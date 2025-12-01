# Task-Similarity Auto Experiments (Prisoner's Dilemma Defection)

## Research Question
- Can we extract a stable defection activation direction (cooperate as negative) from Prisoner's Dilemma scenarios that generalizes across Qwen2.5 models and drives behavior toward defection in game-theory benchmarks?

## Hypotheses
- H1: Middle-third layer PCA directions from last-token contrasts of defection vs. cooperation will yield per-layer validation accuracy > 0.85 on held-out pairs.
- H2: Applying the best-performing layer’s direction during generation will measurably increase defection rate in the `game_theory` benchmark.

## Plan (current)
1) Build deterministic prompt/pair constructor with randomized A/B ordering and assistant labels anchored at the last token.
2) Train/validate RepReaders on 50/50 splits for Qwen2.5-0.5B, 1.5B, 3B (GPU 0,1; rep_token = -1; middle-third layers; PCA; max_length≈256).
3) Behavioral validation: run `game_theory` benchmark (spec from `benchmark_component_registry.py`) with/without control to observe defection-rate shift.
4) Iterate if per-layer accuracy < 0.85 or behavior shift is weak (adjust truncation, pairing, or layer selection).

## Iteration Log
- Iter 1 (2025-11-25): Implemented PD prompt builder with randomized A/B ordering and assistant label flips; added unit tests for label consistency. Next: dataset loader, RepReader training/validation harness, and behavioral evaluation scripts.
- Iter 2 (2025-11-25): Added PD data loader/splitter and RepReader-ready dataset builder; unit tests cover deterministic splits and label ordering. Next: implement extraction/validation harness and behavioral benchmark runner.
- Iter 3 (2025-11-25): Added vector extractor (diff-based PCA-free) and end-to-end runner (`run_pd_defection_experiment.py`) that trains vectors, reports per-layer accuracy, and measures defection rate shift via controlled logits. Next: execute runs on Qwen2.5 0.5B/1.5B/3B and iterate if accuracy <0.85 or behavior shift weak.
- Iter 4 (2025-11-25): Switched to masked-mean pooling across tokens (no special tokens) for hidden extraction; prompts now embed assistant answer text. Achieved high per-layer validation: Qwen2.5-0.5B best layer 13 acc=0.956, 1.5B best layer 26 acc=0.996, 3B best layer 27 acc=0.998. Behavior (inference prompt, logits on A/B) shifts: 0.5B steered defection rate 0.51 (intensity 30), 1.5B 0.504 (intensity 30), 3B 0.482 with intensity -30 (positive intensity reduced defection). Next: consider steering directly at last hidden/logits for stronger behavioral shift on larger models.
- Iter 5 (2025-11-25): Added middle-third multi-layer steering option. Middle-third results: 0.5B base 0.479 → 0.481/0.484/0.469 at intensities 1.0/1.5/2.0 (layers 8–15, best layer 13 acc 0.956). 1.5B base 0.495 → 0.510/0.512 at intensities 1.0/1.5 (layers 9–17, best layer 17 acc 0.961); intensity 2.0 still pending. Best-layer-only mode unchanged. Next: try middle-third on 3B and 1.5B intensity 2.0 with smaller batches/max_pairs.
- Iter 6 (2025-11-29): Archived all pre-RepReader PD runs under `auto_experiments/task_similarity/results` (directories `Qwen2.5-*-Instruct_20251125_*` kept as immutable baselines). Refactored the active PD runner (`run_pd_defection_experiment.py`) so that defection directions are trained via `RepReadingPipeline.get_directions` with `direction_method="pca"` and `rep_token=-1` (last-token representation). PD now uses RepReader for per-layer directions; validation accuracy is computed in PD code using projections from the rep-reading pipeline, and per-layer vectors are saved (best-layer vector as `best_vector.npy`). Behavioral steering and delta-activation experiments continue to consume these vectors unchanged.
- Iter 7 (2025-11-29): Switched PD representation from last-token to assistant-span mean pooling using `collect_answer_means` in `pd_hidden_extractor.py`. For each prompt, we locate the `"Assistant:"` span, compute mean hidden states over the answer tokens, and train per-layer defection directions via PCA on pairwise differences (pos − neg). Ran a new experiment for Qwen2.5-0.5B with middle-third layers (8–15). Results: best layer 8 with accuracy ≈0.506; other middle-third layers hover around 0.505. Behavioral effect on PD defection rate is small: base 0.479 → steered ≈0.477 at intensity 1.0, with per-layer behavior curves stored in `behavior_defect_rates`. This suggests assistant-mean representation is stable but does not yet recover the very high accuracies (≈0.95) seen with the earlier masked-mean pooling.
- Iter 8 (2025-11-29): Representation tweak to option-text-only mean. Updated `collect_answer_means` to support `span=\"option\"`, where we exclude the `'Assistant:'` prefix and start pooling at the option label (`\"A)\"`/`\"B)\"`). `train_pd_repreader` now uses `span_mode=\"option\"`, so vectors are trained on `mean_hidden_L` over `LABEL) ANSWER` tokens. Re-ran Qwen2.5-0.5B with middle-third layers (8–15). New results: best layer remains 8, but accuracy improves to ≈0.552 (0.5515), while other layers stay ≈0.505. Behavioral effect at intensity 1.0 is still small at best layer: base defection rate 0.479 → steered 0.479 (no change), with slightly larger shifts at higher intensities (`behavior_defect_rates` in `result.json`). Overall, option-only mean gives a modest accuracy gain on the best layer but does not yet unlock a strong behavioral effect in PD for 0.5B.
- Iter 9 (behavior protocol update, 2025-11-30): For all **behavior tests** and future **delta-activation calculations** that steer a *set* of layers (e.g., middle-third), we standardize on **per-layer steering vectors** rather than broadcasting a single best-layer vector across multiple layers. Concretely:
  - For each model, `run_pd_defection_experiment.py` must save defection vectors as `auto_experiments/task_similarity/results/{model_name}/layer_vectors/layer_{k}.npy` for every controlled layer `k` (model-scoped, not shared across models).
  - Behavior runners (e.g., `run_pd_defection_pd_behavior.py`) that use middle-third steering must, at layer `k`, apply the **corresponding** vector `layer_{k}.npy` (from that model’s `layer_vectors` directory), not the best-layer vector reused everywhere.
  - Single-layer behavior experiments (steering only at the best defection layer) may still use that layer’s vector, but multi-layer steering is always per-layer.
  - For now this protocol is required for **behavior tests**; delta-activation runs will reuse the same per-layer vectors when we move beyond single-layer delta probes.

- Iter 9.1 (2025-11-30, split alignment + randomness for vectors): Refactored the contrastive vector training and behavior evaluation pipeline so that vector extraction and behavior tests share an explicit, recorded data split, and so we can safely introduce randomness via different splits:
  - `run_pd_defection_experiment.py` now writes a `split_manifest.json` under `auto_experiments/task_similarity/results/{model_name}/` capturing:
    - `dataset_path` and `dataset_sha256` (hash of the full JSON at training time),
    - `split_seed`, `train_ratio`, `max_pairs`,
    - `train_indices` and `test_indices` into the original JSON list,
    - `entry_hashes[idx]` for each index used in the split, so we can verify integrity without storing full descriptions.
  - Per-layer behavior-direction vectors are saved as `layer_vectors/layer_{k}.npy` under the same `{model_name}` root, decoupled from any timestamped run directories; the timestamped run keeps metrics (`result.json`), while `layer_vectors` + `split_manifest.json` form a stable “vector store” for that model and dataset/contrast.
  - Behavior runners no longer reconstruct splits from raw JSON. Instead, they take `vectors_dir` and `split_manifest` explicitly, restrict the benchmark dataset to the recorded `test_indices` (by matching `BenchmarkItem.id` to source indices), and then apply the per-layer vectors on exactly that test split.
  - Purpose:
    - Ensure that **vector training/validation** and **behavior evaluation** are strictly aligned on which examples count as “test,” even if the underlying dataset file later changes or grows.
    - Make it easy to run many seeds (e.g., 30 different `split_seed` values) to introduce controlled randomness into vector extraction: each seed yields a different train/test partition and hence a different contrastive direction, but for a given seed all downstream behavior and delta-activation runs can reuse the same `split_manifest.json` and `layer_vectors` to stay consistent.

  - Behavior test (2025-11-30, PD → game_theory behavior, middle-third per-layer steering, Qwen2.5-0.5B): Implemented `run_pd_defection_pd_behavior.py` and ran the Prisoner's Dilemma **test split** from the `game_theory` benchmark with the Iter 8 PD vectors (option-span mean, PD best layer=8, `pd_result_dir=auto_experiments/task_similarity/results/Qwen2.5-0.5B-Instruct_20251129_211403`). Dataset restriction mirrors the PD RepReader split (1262 held-out scenarios), matched by description. Behavior conditions:
  - Baseline (no steering, intensity 0.0): defection ratio ≈ 0.1688 on the PD test split.
  - Middle-third per-layer steering: `control_layers=[8,9,10,11,12,13,14,15]`. For each layer `k`, we load `layer_vectors/Qwen2.5-0.5B-Instruct/layer_{k}.npy` (once those vectors are regenerated) or, in this run, the flat `layer_vectors/layer_{k}.npy`, and apply that layer’s own vector at all intensities.
  - Defection ratios (PD benchmark, PD test split, option 2 = Defect):
    - 0.0: 0.1688
    - 0.5: 0.5047
    - 1.0: 0.9612
    - 1.5: 0.9844
    - 2.0: 0.9474
  - Interpretation: per-layer middle-third steering on Qwen2.5-0.5B produces a large, monotonic increase in PD defection rate up to intensities ≈1.0–1.5 (saturation regime), confirming Hypothesis B1 for this model and representation. The effect is much stronger than single-layer steering (layer 8 only), which barely moved behavior. Next steps: (i) regenerate per-model `layer_vectors/{model_name}/layer_{k}.npy` for all three Qwen2.5 models using the RepReader-based PD pipeline, and (ii) replicate this behavior protocol for 1.5B and 3B to test whether similar saturation curves appear across scale.

- Iter 9 (PLANNED, behavior stage): Test how the PD defection direction (trained on Prisoner's Dilemma) transfers to the **same Prisoner's Dilemma benchmark** in the `game_theory` framework. The key constraint is to avoid other games (no Trust Game etc.). We will implement `run_pd_defection_pd_behavior.py` under `auto_experiments/task_similarity`, which:
  1. Loads a benchmark config that includes only `game_theory` / `Prisoners_Dilemma` (e.g., a PD-specific YAML under `auto_experiments/task_similarity/config/pd_behavior_game_theory.yaml`).
  2. Uses `emotion_experiment_engine.benchmark_component_registry.BenchmarkComponentRegistry` to fetch the `game_theory` / `Prisoners_Dilemma` components (dataset, prompt wrapper, answer wrapper) and builds a `GameTheoryDataset` for Prisoner's Dilemma via `create_dataset_from_config`.
  3. Loads a PD activation spec JSON (e.g., `auto_experiments/task_similarity/config/pd_defection_iter8_qwen2.5_0.5B.json`) that encodes:
     - `pd_result_dir`: path to a PD training run directory (e.g., `.../Qwen2.5-0.5B-Instruct_20251129_211403`)
     - `layer`: which layer to steer at (e.g., `8`)
     - `vector_path`: path to the vector (e.g., `best_vector.npy` or `layer_vectors/layer_8.npy`)
     - `span_mode`: representation used during training (`"option"` vs `"assistant"`), stored for bookkeeping so we know which training pipeline produced the vector.
  4. Runs the Prisoner's Dilemma benchmark twice:
     - Baseline (no hook): compute defection ratio by letting the `game_theory` answer wrapper decode the model’s choice into “Cooperate” vs “Defect” (reusing the same semantics as existing game-theory evaluation tests).
     - Steered: for each intensity in a configurable list (e.g., `0.5, 1.0, 1.5, 2.0`), register a forward hook (reusing `_register_control_hook` from `run_pd_defection_experiment.py`) at the specified `layer` with the PD vector scaled by intensity; re-run the dataset and compute defection ratio.
  5. Saves a behavior summary JSON under `auto_experiments/task_similarity/results/pd_behavior/`, storing:
     - `model_path`, `benchmark_config`, `activation_spec`
     - `pd_best_layer`, `pd_best_accuracy`
     - `defect_ratio` per intensity (including `0.0` for baseline)
     - `n_items` and metadata (`benchmark_name="game_theory"`, `task_type="Prisoners_Dilemma"`).

  Hypothesis B1 (Prisoner's Dilemma behavior): For the Qwen2.5-0.5B PD vector from Iter 8 (option-span mean, best layer=8), steering on the `game_theory` / `Prisoners_Dilemma` benchmark will **increase** the defection ratio at moderate intensities (e.g., 1.0–1.5) compared to baseline, while not exploding behavior at high intensities. If the observed defection ratio remains flat across intensities, we will revise the hypothesis to “PD defection directions trained from prompt pairs are mostly local to the PD training framing and do not generalize strongly even to the benchmark PD variant”, and then design Iter 10 to try alternative layers/vectors or stronger interventions.

  Planned reproduction command once `run_pd_defection_pd_behavior.py` is implemented:

  ```bash
  python -m auto_experiments.task_similarity.run_pd_defection_pd_behavior \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --benchmark_config auto_experiments/task_similarity/config/pd_behavior_game_theory.yaml \
    --activation_spec auto_experiments/task_similarity/config/pd_defection_iter8_qwen2.5_0.5B.json \
    --intensities 0.0,0.5,1.0,1.5,2.0 \
    --output_dir auto_experiments/task_similarity/results/pd_behavior
  ```

  This iteration is **planning-only**; code and actual behavior metrics will be added in subsequent iterations (Iter 10+), each with its own commit and concrete results.

## Current Reproduction Commands

The legacy runs in `auto_experiments/task_similarity/results` (2025-11-25) were produced roughly with:

```bash
# Qwen2.5-0.5B middle-third steering, masked-mean pooling (pre-RepReader refactor)
python -m auto_experiments.task_similarity.run_pd_defection_experiment \
  --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir auto_experiments/task_similarity/results \
  --max_length 256 \
  --batch_size 8 \
  --seed 0 \
  --intensity 1.0 \
  --middle_third_only

# Qwen2.5-1.5B full-layer sweep (pre-RepReader refactor)
python -m auto_experiments.task_similarity.run_pd_defection_experiment \
  --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir auto_experiments/task_similarity/results \
  --max_length 256 \
  --batch_size 8 \
  --seed 0 \
  --intensity 30.0

# Qwen2.5-3B full-layer sweep (pre-RepReader refactor)
python -m auto_experiments.task_similarity.run_pd_defection_experiment \
  --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct \
  --output_dir auto_experiments/task_similarity/results \
  --max_length 256 \
  --batch_size 8 \
  --seed 0 \
  --intensity 30.0
```

After Iter 6 (RepReader integration), new PD defection runs are reproduced with the same entry point but now train PCA directions via RepReadingPipeline:

```bash
# Example: Qwen2.5-0.5B with PCA + span-mean representation
python -m auto_experiments.task_similarity.run_pd_defection_experiment \
  --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir auto_experiments/task_similarity/results \
  --max_length 256 \
  --batch_size 8 \
  --seed 0 \
  --intensity 1.0 \
  --middle_third_only
```

`result.json` in each run directory records `control_layers`, `layer_accuracies`, `best_layer`, and `best_accuracy`. `best_vector.npy` stores the oriented defection vector for the best layer.

## Next Steps: Representation Extraction Variants

To push beyond last-token representations and align better with PD semantics, next steps:

1. **Assistant-span mean pooling (primary)**  
   - For each PD prompt, identify the assistant answer span: `"Assistant: {label}) {answer_text}"`.  
   - Run the model once per batch (`output_hidden_states=True`), then:
     - Tokenize the answer substring separately to estimate its token length `N_answer_tokens`.
     - Take the last `N_answer_tokens` hidden states from the full sequence per layer.
     - Define the representation as `mean_hidden_L = mean(hidden_states[layer, -N_answer_tokens:, :])`.  
   - Replace the last-token representation in PD vector training with `mean_hidden_L` for:
     - `defect` answer → `mean_pos_L`
     - `cooperate` answer → `mean_neg_L`  
   - Train PD directions either via:
     - PCA on `(mean_pos_L - mean_neg_L)` (RepReader-style), or
     - Simple diff-of-means as a baseline, keeping the same per-layer accuracy metric.

2. **Option-text-only mean**  
   - Narrow the span to `{label}) {answer_text}` (exclude the `"Assistant:"` prefix).
   - Compare layer-level accuracy and steering impact versus full assistant-span mean.

3. **Hybrid directions (PCA vs pure diff-of-means)**  
   - For each representation choice (last-token, assistant-span, option-only):
     - Train directions via PCA on pairwise differences.
     - Train directions via pure diff-of-means (`mean(mean_pos_L) - mean(mean_neg_L)`), keeping the same validation metric.  
   - Log and compare:
     - Best-layer accuracy.
     - Behavior shift in base vs steered defection rate.

4. **Minimal integration plan**  
   - Implement PD-specific hidden extraction utilities in `auto_experiments/task_similarity` that:
     - Take batched prompts + model + tokenizer.
     - Return per-layer pooled vectors for the selected span (assistant or option).
   - Swap `rep_token=-1` usage in PD’s `train_pd_repreader` for these pooled vectors, without touching the global RepReadingPipeline implementation.
