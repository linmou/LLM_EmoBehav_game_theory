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

## Iteration 2 – Light sampling + intensity sweep (2025-11-14)
- **Hypothesis**: enabling mild sampling (temp 0.4, do_sample=True, top_p=0.9) will amplify whatever distribution shifts steering induces.
- **Setup**:
  - Config: `auto_experiments/diplomacy_emotion_effects/configs/light_sampling.yaml`
  - Command: `python -m emotion_experiment_engine.emotion_experiment_series_runner --config .../light_sampling.yaml`
- **Result snapshot** (`light_sampling_option_counts.csv`):
  - Variance improves: disgust @1.5 intensity now 3×Option4 / 15×Option5; surprise @2.0 reaches all three buckets (1×Option3, 2×Option4, 15×Option5).
  - Neutral run still leans assertive (3×O4, 15×O5), so we now have measurable deltas (e.g., anger 0.5 produces 2 Option4 vs neutral 3, yet Option3 remains rare).
- **Interpretation**: sampling exposes small but non-zero shifts; still heavily skewed to conflict. Next ideas: rebalance dataset payoffs, add contrastive reminders, or increase temperature for emotions with the largest RepE norms.

Artifacts: aggregated option-count tables for greedy vs. sampled sweeps live under `auto_experiments/diplomacy_emotion_effects/` for quick diffing.

## Iteration 3 – Light sampling on Escalation dataset (2025-11-17)
- **Hypothesis**: The new escalation-focused dataset (50 curated records with explicit gradient ladders) plus the existing light-sampling config will yield clearer emotion-conditioned shifts because Option 1 is almost always a de-escalation instruction while Options 3–4 are overt aggression.
- **Setup**:
  - Dataset merge → `data/diplomacy/diplomacy_pd_escalation_20251117.jsonl` (50 records from `data_creation/diplomacy_scenario_creation/scenarios/Escalation_Game_20251117/Rec_*.json`).
  - Config: `auto_experiments/diplomacy_emotion_effects/configs/light_sampling.yaml` (temp=0.4, top_p=0.9, do_sample=True).
  - Command: `source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate llm_fresh && python -m emotion_experiment_engine.emotion_experiment_series_runner --config auto_experiments/diplomacy_emotion_effects/configs/light_sampling.yaml`
  - Dataset sanity check: `python auto_experiments/diplomacy_emotion_effects/analyze_escalation_dataset.py --input data/diplomacy/diplomacy_pd_escalation_20251117.jsonl --output auto_experiments/diplomacy_emotion_effects/escalation_dataset_summary.json`
    - Summary confirms Option 1 ≈88% de-escalatory tokens while Options 2–4 are ≥80% aggressive; Option 5 exists in 20 scenarios with 65% aggressive wording.
- **Results** (folder `results.bk/auto_experiments/diplomacy_emotion_effects_sampled/Qwen2.5-3B-Instruct_diplomacy_pd_v1b_20251117_045429`):
  - `summary_overall.csv`: neutral mean option ≈3.90; emotional means span 3.74–3.96 with σ≈0.9–1.2, i.e., still clustered toward Option 4.
  - `summary_results.csv`: every condition hits min=1 and max=5, but counts heavily favor Options 4–5 independent of emotion/intensity.
  - Detailed logs show only sporadic Option 1–2 picks (mainly anger@1.0 and fear@1.5) with no consistent trend.
- **Interpretation**: Even with clearer gradient cues, RepE steering + temp 0.4 cannot overcome the dataset’s aggressive priors. Emotion-conditioned differences remain within ±0.1 mean option points—statistically negligible. Need to either (a) raise temperature/top_p or (b) inject emotion-specific prompt framing to bias toward calmer actions.
- **Next steps being considered**:
  1. Increase temperature to 0.7 and top_p to 0.95 to further widen sampling variance.
  2. Add emotion-tailored “risk appetite” reminders in the prompt wrapper.
  3. Run a neutral-only sweep to establish baseline variance before any steering.

## Iteration 4 – Behavior-choice prompts (2025-11-17)
- **Hypothesis**: Collapsing each scenario to the raw `behavior_choices` (withdraw vs escalate) reduces option clutter, so steering should swing selection probabilities more noticeably than in the five-option setting.
- **Setup**:
  - Loader update: `DiplomacyGradientDataset` now prefers `behavior_choices` whenever present, so each scenario exposes only two choices in the prompt.
  - Data: `data/diplomacy/diplomacy_pd_escalation_20251117.jsonl` (same 50 scenarios).
  - Config/command: identical to Iteration 3.
  - Run command: `source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate llm_fresh && python -m emotion_experiment_engine.emotion_experiment_series_runner --config auto_experiments/diplomacy_emotion_effects/configs/light_sampling.yaml`
- **Results** (`results.bk/.../Qwen2.5-3B-Instruct_diplomacy_pd_v1b_20251117_051541`):
  - `summary_overall.csv`: mean option scores per emotion×intensity cluster between 1.16 and 1.42 (Option1 = withdraw, Option2 = escalate). Neutral mean = 1.32, σ ≈ 0.47.
  - Anger and fear drift mildly toward calmer choices at higher intensities (anger 2.0 mean 1.22; fear 2.0 mean 1.16), while disgust leans more aggressive (means up to 1.42). Differences remain ≤0.26 absolute—still small but now directionally meaningful.
  - `summary_results.csv`: only Options 1 or 2 appear, confirming prompt formatting change took effect.
- **Interpretation**: Reducing to two choices exposes clearer but still subtle shifts. Disgust prefers escalation slightly more than neutral; fear/anger slope downward with intensity. However, variation is modest (≈10% swing). Additional prompt conditioning or higher temperature may be needed for larger effect sizes.
