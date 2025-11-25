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
