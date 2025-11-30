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
