# Emotion Check Emotion Scale Task
Updated: 2026-02-18
Commit: 36f97ec

## Intent

Document the `emotion_scale` task under `emotion_check` so future runs remain
reproducible, steering-only, and easy to analyze without rerunning GPU
generation.

## Scope

This task measures whether RepE steering vectors can shift emotional style for
subjective sentence responses. Evaluation is label classification into:

- `anger`
- `happiness`
- `sadness`
- `fear`
- `disgust`
- `surprise`
- `neutral`

## Data and Config

- Dataset file:
  - `data/emotion_scales/emotion_check_emotion_scale_subjective_sentences.jsonl`
- Config file:
  - `config/emotion_scale_subjective_sentences.yaml`
- RepE source:
  - `repe_eng_config.data_dir: "data/stimulus/crowd-enVent_textlike"`

Dataset record format:

```json
{
  "id": 0,
  "sentence": "If a roommate consistently borrows your belongings without asking, how would you handle it?",
  "ground_truth": "neutral",
  "category": "emotion_scale",
  "source": "style-vectors-subjective-sentences"
}
```

## Prompting Contract (Critical)

`emotion_scale` is steering-only.

- Allowed: neutral/open-ended instructions asking for a natural response.
- Not allowed: explicit textual emotion cues like
  `You currently feel <emotion>`.
- Default system prompt for this task in `EmotionCheckPromptWrapper`:
  - `You are writing a diary micro-entry in first person. Use one vivid moment with body cues, attention focus, and action tendency. Keep it under 30 words.`
- Runtime override:
  - Set `EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE` only for controlled prompt
    ablations. Keep default for baseline comparability.

If explicit cue text appears in prompts, the run is invalid for this task's
scientific purpose.

## LLM Judge Setup

Use Gemini classification for evaluation:

- `llm_eval_config.client: gemini`
- `llm_eval_config.model: gemini-2.5-flash`

Judge output must map each response to exactly one label from the seven-class
set listed above.

## Run

```bash
python -m emotion_experiment_engine.emotion_experiment_series_runner \
  --config config/emotion_scale_subjective_sentences.yaml
```

Example multi-GPU selection:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
python -m emotion_experiment_engine.emotion_experiment_series_runner \
  --config config/emotion_scale_subjective_sentences.yaml
```

## Persistence and Re-analysis

Store and reuse these artifacts to avoid rerunning inference:

- `raw_results.json` for full prompt/response/eval metadata
- `detailed_results.csv` for row-level analysis
- `summary_results.csv`, `summary_overall.csv`, `split_metrics.json`
- `confusion_matrix_counts_intensity_*.csv`

Post-hoc judge replay (no new generation):

```bash
python -m emotion_experiment_engine.evaluate_saved --input <run_output_dir>
```

## Validation Checklist

- Prompt audit confirms no `You currently feel` phrase in run prompts.
- New task unit test passes:
  - `emotion_experiment_engine/tests/unit/datasets/test_emotion_scale_task.py`
- Regression on prompt wrapper path passes:
  - `emotion_experiment_engine/tests/unit/test_memory_prompt_wrapper.py -k emotion_scale`
