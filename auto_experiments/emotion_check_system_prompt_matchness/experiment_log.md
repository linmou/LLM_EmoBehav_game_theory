# Emotion Check System Prompt Matchness Log

- Created: 2026-02-19
- Branch: `auto_experiment/system_prompt_matchness`
- Intent: improve steering-to-output emotional match by changing only the `emotion_check` system prompt, without explicit emotion wording.

## Research Questions
- Can system prompt framing increase steered plain accuracy and match score on `psyset_emotion_eval`?
- Which prompt characteristics help most: concreteness, diary framing, appraisal/action framing, or instinctive phrasing?
- Does improvement hold only at high intensity or also across a full intensity sweep?

## Baseline Reference
- Existing baseline run (no extra rerun): `results/psyset_emotion_eval/crowd-enVent_textlike/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_160043`
- Baseline steered plain accuracy: `0.2387`
- Baseline steered match score: `0.2129`

## Plan
1. Add runtime system-prompt override hook + test (no code edits between prompt variants).
2. Run quick automatic prompt search at intensity `4.0`.
3. Select best prompt and run full intensity sweep (`0.5` to `5.0`).
4. Compare against baseline and log accepted/rejected hypotheses.

## Iteration Log

### Iteration 1 (completed)
- Hypothesis: the current system prompt is misaligned with free-text PsySET output; a concise first-person style prompt without emotion words will improve matchness.
- Setup:
  - Config: `auto_experiments/emotion_check_system_prompt_matchness/configs/quick_intensity4.yaml`
  - Runner: `auto_experiments/emotion_check_system_prompt_matchness/run_prompt_variant_search.py`
  - Variants: `auto_experiments/emotion_check_system_prompt_matchness/prompt_variants_iteration1.json`
  - Command:
    - `python auto_experiments/emotion_check_system_prompt_matchness/run_prompt_variant_search.py --variants-json auto_experiments/emotion_check_system_prompt_matchness/prompt_variants_iteration1.json --cuda-devices 2,3`
- Results summary: `auto_experiments/emotion_check_system_prompt_matchness/quick_prompt_search_summary.csv`
  - baseline: accuracy `0.2571`, match score `0.2334`
  - v1_concrete_state: accuracy `0.3429`, match score `0.3031`
  - v2_diary_micro: accuracy `0.4048`, match score `0.3612` (best)
  - v3_appraisal_action: accuracy `0.3143`, match score `0.2911`
  - v4_instinctive_reaction: accuracy `0.3000`, match score `0.2653`
- Hypothesis check:
  - Supported. Non-emotion-worded system prompt framing materially increased steered matchness.
  - Best variant (`v2_diary_micro`) outperformed baseline by `+0.1476` plain accuracy and `+0.1278` match score at intensity `4.0`.

### Iteration 2 (planned)
- Hypothesis: the best prompt from Iteration 1 (`v2_diary_micro`) will improve full-sweep metrics across intensities `0.5` to `5.0`, not only at a single high intensity.
- Setup:
  - Config: `auto_experiments/emotion_check_system_prompt_matchness/configs/full_sweep_best_prompt.yaml`
  - Override env:
    - `EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE="You are writing a diary micro-entry in first person. Use one vivid moment with body cues, attention focus, and action tendency. Keep it under 30 words."`
  - Command:
    - `CUDA_VISIBLE_DEVICES=2,3 /home/jjl7137/anaconda3/bin/conda run -n llm python -m emotion_experiment_engine.emotion_experiment_series_runner --config auto_experiments/emotion_check_system_prompt_matchness/configs/full_sweep_best_prompt.yaml`
