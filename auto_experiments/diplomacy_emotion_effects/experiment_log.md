# Diplomacy Emotion Effects Experiment Log

- Created: 2025-11-14T02:08:22-05:00
- Purpose: Investigate why activation steering did not change Diplomacy choices; search for settings that surface effects.

Research Questions
- Do multi-layer steering (middle third) + intensity sweep change option distributions across emotions?
- Does light sampling reveal effects hidden by greedy decoding?

Method Outline
- Benchmark: diplomacy_pd (JSONL v1b).
- Model: Qwen2.5-3B-Instruct.
- Emotions: anger, sadness, happiness, fear, disgust, surprise (+ neutral baseline).
- Intensities: 0.5, 1.0, 1.5, 2.0.
- Decoding: greedy vs light sampling.

Iteration Log
- Iteration 1 (plan): Use middle-1/3 layers (already in code), copy dataset to v1b adding ordering hint to scenario; run greedy with intensity sweep.
- Iteration 2 (plan): Same but with light sampling (temp=0.4, top_p=0.9).

## Iteration 1 – Greedy decoding + intensity sweep (2025-11-14)
- **Hypothesis**: steering across the middle third of layers plus an intensity sweep (0.5–2.0) will create visible shifts in option selection, even under greedy decoding.
- **Setup**:
  - Config: `auto_experiments/diplomacy_emotion_effects/configs/greedy.yaml`
  - Command: `python -m emotion_experiment_engine.emotion_experiment_series_runner --config .../greedy.yaml`
  - Notes: Had to recreate Diplomacy dataset/prompt modules, register them, stub `api_configs`, and symlink `data/stimulus` to the canonical dataset so RepE readers could build.
- **Result snapshot** (`greedy_option_counts.csv`):
  - Most emotions stay pinned to `Option 5` (aggressive) regardless of intensity (e.g., anger: 16/18 selections at each intensity, only 1 sample dipping to Option 3/4).
  - Disgust/fear show minor movement toward Option 4 but still 14+ selections at Option 5.
  - Neutral baseline mirrors the emotional runs (16×Option5, 2×Option4).
- **Interpretation**: steering alone can’t overcome deterministic decoding; logits shift but argmax remains stuck near aggressive moves. Need stochastic decoding to surface signal.

Artifacts: aggregated option-count tables for the greedy sweep live under `auto_experiments/diplomacy_emotion_effects/` for quick diffing.
