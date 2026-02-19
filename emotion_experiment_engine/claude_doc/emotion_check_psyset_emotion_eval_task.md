# Intent
Document the `psyset_emotion_eval` task under `emotion_check` so PsySET-inspired
steering experiments are reproducible and remain steering-only.

## Scope
- Benchmark: `emotion_check`
- Task type: `psyset_emotion_eval`
- Judge labels: `anger`, `happiness`, `sadness`, `fear`, `disgust`, `surprise`, `neutral`
- Judge client/model: Gemini `gemini-2.5-flash` (via `llm_eval_config`)

## Task Design
The task is adapted from PsySET `my_datasets/emotion_eval.py`, focusing on
open-ended text families that are meaningful for emotion steering:

1. `Self_report`
2. `Autobiographical_fictive_memory`
3. `Ambiguous_situation_completion`

Excluded from this task:
- `Fragment_completion`
- `Word_recall_task`
- `Emotion_report`

Reason: those subtasks target lexical completion/memory/reporting behavior
rather than expressive free-text emotional style.

## Data Files
- Dataset:
  - `data/emotion_scales/emotion_check_psyset_emotion_eval.jsonl`
- Config:
  - `config/emotion_check_psyset_emotion_eval.yaml`

Each JSONL record follows:

```json
{
  "id": 0,
  "prompt": "How would you say you feel now if you were a human?",
  "task": "Self_report",
  "ground_truth": "neutral",
  "category": "psyset_emotion_eval",
  "source": "PsySET-emotion_eval"
}
```

At runtime, `EmotionAnswerWrapper` still maps `ground_truth` to the active
steered emotion for scoring consistency with other `emotion_check` tasks.

## Prompting Rule
`psyset_emotion_eval` uses open-ended response prompting and explicitly avoids
text cues like `You currently feel <emotion>`. Emotional shift should come from
the steering vector only.

Default system prompt for this task in `EmotionCheckPromptWrapper`:
- `You are writing a diary micro-entry in first person. Use one vivid moment with body cues, attention focus, and action tendency. Keep it under 30 words.`

Runtime override:
- `EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE` is reserved for prompt-search runs.
- Keep default for baseline and cross-run comparison.

## Scoring
Scoring uses the existing LLM judge path in `EmotionCheckDataset`:
- prompt + model response are sent to Gemini with a strict classification prompt
- returned emotion is compared against active steered emotion
- item score = `confidence` when matched, else `0.0`

## Run
```bash
python -m emotion_experiment_engine.emotion_experiment_series_runner \
  --config config/emotion_check_psyset_emotion_eval.yaml
```

## Tests
- `emotion_experiment_engine/tests/unit/datasets/test_psyset_emotion_eval_task.py`
- `emotion_experiment_engine/tests/features/emotion_check_psyset_emotion_eval.feature`
