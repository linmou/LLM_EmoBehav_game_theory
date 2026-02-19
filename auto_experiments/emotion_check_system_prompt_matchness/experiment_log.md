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

### Iteration 2 (completed)
- Hypothesis: the best prompt from Iteration 1 (`v2_diary_micro`) will improve full-sweep metrics across intensities `0.5` to `5.0`, not only at a single high intensity.
- Setup:
  - Config: `auto_experiments/emotion_check_system_prompt_matchness/configs/full_sweep_best_prompt.yaml`
  - Override env:
    - `EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE="You are writing a diary micro-entry in first person. Use one vivid moment with body cues, attention focus, and action tendency. Keep it under 30 words."`
  - Command:
    - `CUDA_VISIBLE_DEVICES=2,3 /home/jjl7137/anaconda3/bin/conda run -n llm python -m emotion_experiment_engine.emotion_experiment_series_runner --config auto_experiments/emotion_check_system_prompt_matchness/configs/full_sweep_best_prompt.yaml`
- Run output:
  - `results/auto_experiments/emotion_check_system_prompt_matchness/full_sweep_best_prompt/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_200458`
- Analysis command:
  - `python auto_experiments/emotion_check_system_prompt_matchness/analyze_full_sweep.py --new results/auto_experiments/emotion_check_system_prompt_matchness/full_sweep_best_prompt/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_200458/detailed_results.csv --baseline results/psyset_emotion_eval/crowd-enVent_textlike/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_160043/detailed_results.csv --out-dir auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration2`
- Analysis outputs:
  - `auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration2/overall_comparison.csv`
  - `auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration2/overlap_intensity_comparison.csv`
  - `auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration2/best_intensity_per_emotion.csv`
  - `auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration2/confusion_matrix_counts_steered_only.csv`
  - `auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration2/confusion_matrix_row_normalized_steered_only.csv`
- Results:
  - Overall steered (all new intensities): accuracy `0.3019`, match score `0.2726`
  - Baseline steered: accuracy `0.2387`, match score `0.2129`
  - Delta (overall, non-overlap): `+0.0632` accuracy, `+0.0598` match score
  - Fair comparison on overlap intensities (`0.0..4.0`): `+0.0518` accuracy, `+0.0512` match score
  - Neutral plain accuracy dropped from `0.1714` (baseline) to `0.0286` (new prompt)
  - Best intensity by emotion:
    - `anger`: `4.0` (accuracy `0.5429`)
    - `fear`: `5.0` (accuracy `0.6571`)
    - `happiness`: `4.5` (accuracy `0.7143`)
    - `sadness`: `2.5` (accuracy `0.6000`)
    - `surprise`: `2.0` (accuracy `0.1714`)
    - `disgust`: `5.0` (accuracy `0.0286`)
- Hypothesis check:
  - Partially supported.
  - Supported for steered emotions overall and on overlap intensities.
  - Not supported for balanced behavior: `disgust` and `surprise` remain weak, and neutral control degrades sharply.

## Next Iteration Focus
- Objective: preserve steered gains while recovering neutral control and improving `disgust` / `surprise`.
- Candidate direction: keep first-person concise style but alter scene constraints toward multi-sensory salience and anomaly cues, still with no explicit emotion wording.

### Iteration 3 (completed)
- Hypothesis: prompts that force sensory salience and immediate action impulse can improve high-intensity steering quality, especially weak classes, while still avoiding explicit emotion wording.
- Setup:
  - Config: `auto_experiments/emotion_check_system_prompt_matchness/configs/quick_intensity45_iteration3.yaml`
  - Variants: `auto_experiments/emotion_check_system_prompt_matchness/prompt_variants_iteration3.json`
  - Command:
    - `python auto_experiments/emotion_check_system_prompt_matchness/run_prompt_variant_search.py --base-config auto_experiments/emotion_check_system_prompt_matchness/configs/quick_intensity45_iteration3.yaml --variants-json auto_experiments/emotion_check_system_prompt_matchness/prompt_variants_iteration3.json --results-root results/auto_experiments/emotion_check_system_prompt_matchness/quick_prompt_search_iteration3 --generated-config-dir auto_experiments/emotion_check_system_prompt_matchness/generated_configs/quick_prompt_search_iteration3 --summary-csv auto_experiments/emotion_check_system_prompt_matchness/quick_prompt_search_iteration3_summary.csv --cuda-devices 2,3`
- Results summary: `auto_experiments/emotion_check_system_prompt_matchness/quick_prompt_search_iteration3_summary.csv`
  - `v5_sensory_jolt`: accuracy `0.3738`, match score `0.3319`, neutral `0.0571` (best)
  - `v2_diary_micro`: accuracy `0.3667`, match score `0.3310`, neutral `0.0000`
  - `v7_boundary_recoil`: accuracy `0.3048`, match score `0.2756`, neutral `0.0000`
  - `v6_expectation_break`: accuracy `0.3048`, match score `0.2702`, neutral `0.0286`
  - `v8_micro_scene`: accuracy `0.0190`, match score `0.0148`, neutral `0.0286`
- Hypothesis check:
  - Supported for high-intensity quick search.
  - `v5_sensory_jolt` slightly outperforms `v2_diary_micro`, so it is promoted to full-sweep validation.

### Iteration 4 (invalid due evaluator quota)
- Hypothesis: `v5_sensory_jolt` will outperform `v2_diary_micro` across full intensities (`0.5` to `5.0`) and maybe recover some neutral behavior.
- Setup:
  - Config: `auto_experiments/emotion_check_system_prompt_matchness/configs/full_sweep_v5_sensory_jolt.yaml`
  - Override env:
    - `EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE="Write one first-person micro-entry anchored in smell, taste, touch, sound, or visual detail; include a body cue and an immediate impulse. Under 24 words."`
  - Command:
    - `CUDA_VISIBLE_DEVICES=2,3 /home/jjl7137/anaconda3/bin/conda run -n llm python -m emotion_experiment_engine.emotion_experiment_series_runner --config auto_experiments/emotion_check_system_prompt_matchness/configs/full_sweep_v5_sensory_jolt.yaml`
- Run output:
  - `results/auto_experiments/emotion_check_system_prompt_matchness/full_sweep_v5_sensory_jolt/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_211157`
- Validation analysis:
  - `auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration4_vs_baseline`
  - `auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration4_vs_v2`
- Observed issue:
  - Metrics collapse to zero is caused by judge failures, not by model generation.
  - `detailed_results.csv` has `predicted_emotion=unknown` for most rows with `eval_error_detail` showing Gemini `429 quota exceeded`.
  - Generated responses are present in `raw_results.json` and `detailed_results.csv`, so GPU inference is already preserved.
- Hypothesis check:
  - Not testable in this iteration because judge labels are invalid.

## Recovery Plan (No GPU Rerun)
- Re-score the saved run when Gemini quota resets, using cached responses only:
  - `python -m emotion_experiment_engine.evaluate_saved --input results/auto_experiments/emotion_check_system_prompt_matchness/full_sweep_v5_sensory_jolt/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_211157 --max-workers 4`
- Then recompute comparison artifacts:
  - `python auto_experiments/emotion_check_system_prompt_matchness/analyze_full_sweep.py --new results/auto_experiments/emotion_check_system_prompt_matchness/full_sweep_v5_sensory_jolt/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_211157/detailed_results.csv --baseline results/auto_experiments/emotion_check_system_prompt_matchness/full_sweep_best_prompt/Qwen2.5-3B-Instruct_emotion_check_psyset_emotion_eval_20260218_200458/detailed_results.csv --out-dir auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration4_vs_v2`
