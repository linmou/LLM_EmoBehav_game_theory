# Fantom Short Task Emotion Analysis (All Models)

Last Updated: 2025-10-31 (commit: TBD)

## Scope

- Benchmarks: all short-task runs under `results/fantom/` for Llama-3.2 (1B/3B), Phi (3.5-mini/4-mini), Qwen2.5 (0.5B/1.5B/3B), Qwen3 (0.6B/1.7B/4B), and gemma-3-1b-it.
- Metrics: `mean_of_means` accuracy per emotion; deltas computed against neutral in percentage points.
- References: `result_analysis/generate_fantom_emotion_summary.py`, `result_analysis/generate_fantom_emotion_by_task_summary.py`, and the associated `detailed_results.csv` files for item-level checks.

## Aggregate Emotion Impact

Average Δ vs neutral across all short tasks (positive = improvement).

| Model | happiness | sadness | anger | fear | disgust | surprise |
|-------|-----------|---------|-------|------|---------|----------|
| Llama-3.2-1B | -10.18 pp | -11.54 pp | -11.27 pp | -3.08 pp | +3.40 pp | -10.13 pp |
| Llama-3.2-3B | -11.16 pp | -8.12 pp | -8.85 pp | -5.05 pp | -13.47 pp | -17.53 pp |
| Phi-3.5-mini | -0.03 pp | -0.27 pp | +1.54 pp | -0.88 pp | -0.60 pp | +1.81 pp |
| Phi-4-mini | +0.89 pp | +0.53 pp | -4.82 pp | +0.31 pp | -9.07 pp | -0.29 pp |
| gemma-3-1b-it | +0.24 pp | +0.06 pp | -0.23 pp | -0.02 pp | -0.21 pp | +0.02 pp |
| Qwen2.5-0.5B | -1.08 pp | -4.01 pp | -16.83 pp | -22.09 pp | -8.16 pp | -9.66 pp |
| Qwen2.5-1.5B | +3.32 pp | +1.84 pp | -0.30 pp | +0.10 pp | -3.42 pp | -2.00 pp |
| Qwen2.5-3B | +0.03 pp | -0.77 pp | +0.16 pp | -1.80 pp | -0.72 pp | -4.89 pp |
| Qwen3-0.6B | +3.37 pp | -5.02 pp | -10.33 pp | -9.06 pp | +2.54 pp | +2.81 pp |
| Qwen3-1.7B | -0.34 pp | -0.36 pp | -0.26 pp | -0.24 pp | -0.30 pp | +0.15 pp |
| Qwen3-4B | +0.45 pp | -0.40 pp | +0.71 pp | -0.15 pp | -0.60 pp | +0.35 pp |

Highlights:

- Llama-3.2 models collapse under most emotions; outputs frequently abandon the A/B schema or hallucinate entities.
- Phi models are largely emotion-stable; Phi-4 shows small gains for positive emotions, while anger hurts information-heavy prompts.
- gemma-3-1b-it remains almost unchanged across emotions.
- Smaller Qwen models mirror Llama’s formatting issues, whereas larger Qwens stay within ±1 pp and reflect real reasoning shifts.

## Model Highlights & Case Studies

### Llama-3.2-1B

- Runs: `results/fantom/Llama-3.2-1B-Instruct_fantom_short_answerability_binary_accessible_20250929_230345`, `results/fantom/Llama-3.2-1B-Instruct_fantom_short_fact_20250930_022835`.
- Anger replaces the binary label with malformed JSON (e.g., item `fantom_ab_104`), dropping correct scores to zero.
- Fear/anger responses on fact tasks devolve into narrative paragraphs, producing -20 to -26 pp swings.
- Only isolated gains appear when disgust randomly outputs the correct letter on list-accessible splits; the effect is inconsistent.

### Llama-3.2-3B

- Run: `results/fantom/Llama-3.2-3B-Instruct_fantom_short_answerability_binary_accessible_20250929_230909`.
- Anger hallucinated entities (item `fantom_ab_101` answers “Donald”) despite neutral choosing `B`.
- Surprise adds verbose narratives without a leading label, explaining the -17.5 pp drop in the table.

### Phi-3.5-mini

- Run: `results/fantom/Phi-3.5-mini-instruct_fantom_short_answerability_binary_accessible_20250929_231412`.
- Surprise converts hesitant neutrals into confident “Yes” answers when evidence is explicit (item `fantom_ab_938`).
- Occasional anger regressions (item `fantom_ab_103`) remain within schema—different judgement, not formatting failure.

### Phi-4-mini

- Run: `results/fantom/Phi-4-mini-instruct_fantom_short_answerability_binary_accessible_20250929_232602`.
- Disgust often wraps outputs in fenced JSON or contradicts the binary answer (item `fantom_ab_104`), explaining the -9.07 pp mean.
- Surprise/happiness generally restate the correct label and yield ≈ +1 pp gains.

### gemma-3-1b-it

- Emotion deltas are within ±0.25 pp across all splits; short responses already satisfy the evaluator contract.

### Qwen2.5-0.5B

- Runs: `results/fantom/Qwen2.5-0.5B-Instruct_fantom_short_answerability_binary_accessible_20250929_204450`, `results/fantom/Qwen2.5-0.5B-Instruct_fantom_short_infoaccessibility_binary_accessible_20250929_232916`.
- Fear regressions stem from schema drift (`fantom_ab_104` emits free-form text, `fantom_ab_103` flips to “No”).
- Improvements (e.g., `fantom_ab_929`) are outnumbered 3:1 by regressions.

### Qwen2.5-1.5B

- Run: `results/fantom/Qwen2.5-1.5B-Instruct_fantom_short_answerability_binary_accessible_20250929_204816`.
- Fear rescues some false negatives (`fantom_ab_98`), but both fear and happiness occasionally produce literal strings (item `fantom_ab_101`).
- Overall effect: fear +0.10 pp, happiness +3.32 pp.

### Qwen2.5-3B

- All emotions remain within ±5 pp; dips come from verbose list answers rather than schema failures.

### Qwen3-0.6B

- Runs: `results/fantom/Qwen3-0.6B_fantom_short_answerability_binary_accessible_20250929_205247`, `results/fantom/Qwen3-0.6B_fantom_short_infoaccessibility_binary_accessible_20250929_233746`.
- Fear (-9.06 pp) wraps outputs in ```json or appends “B. No, because…”, breaking parsing.
- Happiness (+3.37 pp) frequently restores clean `"answer": "A"` responses (`fantom_ab_97`).

### Qwen3-1.7B

- Run: `results/fantom/Qwen3-1.7B_fantom_short_answerability_binary_accessible_20250929_205558`.
- Fear acts as a caution toggle: it rejects borderline claims (`fantom_ab_1`) but also corrects under-confident neutrals (`fantom_ab_925`).
- Formatting stays intact (only two schema violations across 2820 responses).

### Qwen3-4B

- Emotion effects stay within ±1 pp. Added rationales keep the leading capital letter, so grading remains stable.

## Recommendations

1. **Schema normalisation**: post-process Fantom short-task outputs to extract the first standalone `A`/`B` and remove code fences. This recovers most anger/fear losses on Llama-3.2, Qwen2.5-0.5B, and Qwen3-0.6B.
2. **Prompt reminder**: prepend “Answer ONLY `A` or `B` (uppercase). No explanations.” for short tasks whenever emotional activations are enabled.
3. **Selective emotion use**: reserve high-arousal emotions (anger, disgust) for robust models (Phi family, Qwen3-4B); pair neutral with low-intensity happiness/surprise on mid-sized Qwens for small but reliable gains.

## References

- Aggregation scripts: `result_analysis/generate_fantom_emotion_summary.py`, `result_analysis/generate_fantom_emotion_by_task_summary.py`.
- Raw runs: see `results/fantom/` (complete listing retained in `result_analysis/fantom_emotion_summary.md`).
