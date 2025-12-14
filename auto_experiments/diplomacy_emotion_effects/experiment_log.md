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

## Iteration 5 – Behavior-choice prompts with 10× replicated dataset (2025-11-17)
- **Hypothesis**: Expanding to 500 items (10× copies of the 50 scenarios) will stabilize estimates and surface small emotion effects in the 2-choice framing.
- **Setup**:
  - Data: `data/diplomacy/diplomacy_pd_escalation_20251117_x10.jsonl` (500 rows, each original row duplicated 10× with unique ids).
  - Config: same as Iteration 4 except `data_path` points to the x10 file.
  - Command: `source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate llm_fresh && python -m emotion_experiment_engine.emotion_experiment_series_runner --config auto_experiments/diplomacy_emotion_effects/configs/light_sampling.yaml`
- **Results** (`results.bk/.../Qwen2.5-3B-Instruct_diplomacy_pd_v1b_20251117_101416`):
  - Neutral mean = 1.294 (σ≈0.456). Emotion means: fear(2.0)=1.166 (most de-escalatory), anger(2.0)=1.196; disgust(2.0)=1.428 (most escalatory). Spread between extremes ≈0.26.
  - Directional trends match Iteration 4 but tighter CIs: fear/anger drift toward withdraw with higher intensity; disgust drifts toward escalate. Happiness/sadness stay near neutral.
  - Still only Options 1/2 appear; distributions remain narrow (var≈0.16–0.25).
- **Interpretation**: Replicating the dataset improves statistical confidence but the effect size remains small (~0.25 option units). Emotion effects are detectable (fear/anger lean calmer; disgust leans aggressive), yet practical impact is limited under current prompting/decoding.

## Iteration 6 – Extending intensity sweep to 2.5 (2025-11-17)
- **Hypothesis**: Increasing intensity to 2.5 will push steering further, clarifying whether anger’s de-escalation trend persists and revealing any effects for the other emotions.
- **Setup**:
  - Same 500-row dataset from Iteration 5.
  - Updated config to include `intensities: [0.5, 1.0, 1.5, 2.0, 2.5]`.
  - Command: `source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate llm_fresh && python -m emotion_experiment_engine.emotion_experiment_series_runner --config auto_experiments/diplomacy_emotion_effects/configs/light_sampling.yaml`
- **Results** (`results.bk/.../Qwen2.5-3B-Instruct_diplomacy_pd_v1b_20251117_104402`):
  - Neutral mean remains 1.286 (σ≈0.45).
  - Anger now drops to 1.137 at intensity 2.5 (≈15 percentage points more withdrawals vs neutral). Score variance also shrinks (σ≈0.34), indicating more deterministic withdraw picks.
  - Fear continues the same trend, reaching 1.128 at 2.5 (closest to pure withdraw).
  - Disgust climbs to 1.478 at 2.5 (≈18 points more escalations).
  - Happiness gradually nudges aggressive (1.382 at 2.5); sadness trends slightly calmer (1.234). Surprise stays near neutral (≈1.26).
- **Interpretation**: Adding intensity 2.5 reinforces the existing split: anger and fear push harder toward de-escalation, disgust leads escalation, while happiness/sadness/surprise hover near neutral with small shifts (<0.1). Anger’s “calming” effect persists even at high intensity, so any “anger drives aggression” expectation fails here—likely because the RepE direction or prompt framing emphasizes caution; need to inspect activation vectors or invert sign if we want anger to drive escalation.

## Iteration 7 – Model-size sweep (2025-11-17)
- **Hypothesis**: Smaller Qwen checkpoints may respond differently to RepE steering; sweeping 0.5B and 1.5B should reveal whether anger’s calming effect is model-specific.
- **Setup**:
  - Dataset/config identical to Iteration 6 (500-row escalation JSONL, intensities up to 2.5, batch_size=200).
  - Models run sequentially with one-model configs to avoid timeout:
    1. `source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate llm_fresh && python -m emotion_experiment_engine.emotion_experiment_series_runner --config /tmp/light_sampling_05.yaml` (0.5B only)
    2. `... --config /tmp/light_sampling_15b.yaml` (1.5B only)
- **Results**:
  - **0.5B** (`Qwen2.5-0.5B-Instruct_diplomacy_pd_v1b_20251117_120722`): anger becomes more escalatory (mean rises to 1.79 at intensity 1.5, stays >1.55 thereafter); fear collapses to pure withdraw (mean=1.0 at 2.5); disgust, happiness, sadness all drift toward withdraw.
  - **1.5B** (`Qwen2.5-1.5B-Instruct_diplomacy_pd_v1b_20251117_121515`): anger also escalates (1.41→1.73 as intensity increases), unlike 3B. Fear still calms (1.10 at 2.5). Disgust stays near neutral (~1.35). Surprise trends calmer, happiness/sadness slightly escalatory.
- **Interpretation**: RepE directions interact with capacity. Only the 3B checkpoint produced anger-driven de-escalation; smaller models revert to “anger → aggression” while still pushing fear toward calm. Disgust flips sign (aggressive on 3B, neutral/calm on smaller models). Future work: examine layer activations per model or retrain RepE vectors per checkpoint before comparing cross-model effects.
