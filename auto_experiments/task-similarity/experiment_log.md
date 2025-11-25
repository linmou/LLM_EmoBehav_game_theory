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
