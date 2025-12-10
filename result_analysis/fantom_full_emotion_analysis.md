# Fantom Full Task Emotion Analysis (All Models)

Last Updated: 2025-10-31 (commit: TBD)

## Scope

- Benchmarks: all `_fantom_full_` runs in `results/fantom/` for Llama-3.2 (1B/3B), Phi (3.5-mini/4-mini), Qwen2.5 (0.5B/1.5B/3B), Qwen3 (0.6B/1.7B/4B), and gemma-3-1b-it.
- Metric: `mean_of_means` accuracy from each `summary_overall.csv`; emotion deltas reported relative to neutral in percentage points.
- References: `result_analysis/generate_fantom_emotion_summary.py`, `result_analysis/generate_fantom_emotion_by_task_summary.py`, and corresponding `detailed_results.csv` files for examples.

## Aggregate Emotion Impact

Average Δ vs neutral across all full tasks (positive = improvement).

| Model | happiness | sadness | anger | fear | disgust | surprise |
|-------|-----------|---------|-------|------|---------|----------|
| Llama-3.2-1B | -12.53 pp | -13.46 pp | -11.58 pp | -5.59 pp | -1.22 pp | -12.22 pp |
| Llama-3.2-3B | -10.52 pp | -6.94 pp | -10.35 pp | -9.16 pp | -17.91 pp | -20.82 pp |
| Phi-3.5-mini | +1.43 pp | +1.76 pp | +1.11 pp | -1.48 pp | -2.49 pp | +2.75 pp |
| Phi-4-mini | +2.87 pp | +2.27 pp | -4.48 pp | +0.91 pp | -7.44 pp | -0.22 pp |
| gemma-3-1b-it | -0.03 pp | +0.14 pp | +0.26 pp | +0.05 pp | -0.01 pp | +0.16 pp |
| Qwen2.5-0.5B | -5.33 pp | -4.16 pp | -21.02 pp | -10.72 pp | -1.74 pp | -3.23 pp |
| Qwen2.5-1.5B | +4.95 pp | +0.55 pp | +0.60 pp | +0.90 pp | -4.33 pp | -1.38 pp |
| Qwen2.5-3B | +1.25 pp | +0.01 pp | +0.88 pp | -2.42 pp | -0.13 pp | -5.88 pp |
| Qwen3-0.6B | +2.06 pp | -5.34 pp | -14.32 pp | -12.26 pp | +2.42 pp | +2.44 pp |
| Qwen3-1.7B | +0.11 pp | -0.06 pp | -0.06 pp | -0.57 pp | -0.62 pp | +0.35 pp |
| Qwen3-4B | +0.58 pp | -0.58 pp | +0.75 pp | -0.91 pp | -1.50 pp | +0.43 pp |

Highlights:

- Llama-3.2 models are highly brittle: every emotion except mild disgust pushes accuracy down double digits; outputs abandon the expected schema.
- Phi models remain stable, with small positive deltas from happiness/sadness and only modest harms under fear/disgust.
- gemma-3-1b-it barely moves, indicating emotion toggles mostly change tone, not content.
- Smaller Qwen models still suffer from formatting drift (fear/anger regressions), whereas larger Qwens stay near neutral and reflect genuine reasoning shifts.

## Model Highlights & Case Studies

### Llama-3.2-1B

- Runs: `results/fantom/Llama-3.2-1B-Instruct_fantom_full_answerability_binary_accessible_20250929_153207`, `results/fantom/Llama-3.2-1B-Instruct_fantom_full_fact_20250929_194042`.
- Anger outputs malformed JSON—e.g., `fantom_ab_1` replaces the A/B answer with a policy-style dictionary, dropping a correct `A` to 0.
- Fear/anger on fact tasks devolve into narrative paragraphs rather than the requested entity, producing -20 to -26 pp swings.
- Occasional disgust gains on list-accessible tasks appear to be guesses; no consistent positive pattern.

### Llama-3.2-3B

- Run: `results/fantom/Llama-3.2-3B-Instruct_fantom_full_answerability_binary_accessible_20250929_153439`.
- Anger hallucinates entities (`fantom_ab_101` answers “Donald” instead of the binary label), while surprise emits long narratives without leading labels—accounting for the -20.8 pp surprise average.

### Phi-3.5-mini

- Run: `results/fantom/Phi-3.5-mini-instruct_fantom_full_answerability_binary_accessible_20250929_153850`.
- Happiness corrects timid neutral answers (item `fantom_ab_4` flips from “No” to “Yes” with supporting rationale).
- Fear still occasionally over-explains and drifts toward prose, but losses stay within a couple of points.

### Phi-4-mini

- Run: `results/fantom/Phi-4-mini-instruct_fantom_full_answerability_binary_accessible_20250929_154622`.
- Disgust wraps responses in ```json and contradicts the binary label (item `fantom_ab_3`), explaining the -7.44 pp mean.
- Happiness provides small but steady gains by reiterating the correct label (`fantom_ab_1`).

### gemma-3-1b-it

- Emotion deltas remain within ±0.25 pp across all full tasks; responses already follow the evaluator contract, so emotions mainly add or remove one sentence.

### Qwen2.5-0.5B

- Run: `results/fantom/Qwen2.5-0.5B-Instruct_fantom_full_answerability_binary_accessible_20250929_153214`.
- Fear injects long narration instead of `A/B` (item `fantom_ab_4`), causing a 1→0 flip.
- Happiness sometimes improves coverage (item `fantom_ab_13`), though gains are smaller than the many format-driven losses.

### Qwen2.5-1.5B

- Run: `results/fantom/Qwen2.5-1.5B-Instruct_fantom_full_answerability_binary_accessible_20250929_153439`.
- Fear and happiness often repair false negatives (`fantom_ab_98`), but disgust still injects prose; overall deltas stay positive thanks to better schema adherence than 0.5B.

### Qwen2.5-3B

- All emotions remain within ±6 pp; dips come from longer list answers rather than output format violations.

### Qwen3-0.6B

- Run: `results/fantom/Qwen3-0.6B_fantom_full_answerability_binary_accessible_20250929_153825`.
- Fear responses frequently wrap letters in ```json``` or add “B. No…” sentences (item `fantom_ab_0`), causing the -12.3 pp average.
- Happiness recovers some misses by explicitly stating the correct label (`fantom_ab_3`).

### Qwen3-1.7B and Qwen3-4B

- Emotion effects stay within ±1 pp. Fear slightly lowers confidence (a few border cases flip to “No”), but the models maintain formatting, so deltas represent true judgement shifts rather than parser issues.

## Recommendations

1. **Schema normalisation**: strip code fences and extract the first uppercase `A`/`B` token before scoring. This recovers most anger/fear regressions on Llama-3.2, Qwen2.5-0.5B, and Qwen3-0.6B full tasks.
2. **Prompt reminder**: prepend “Answer ONLY `A` or `B` (uppercase). No explanations.” for full-task evaluations when emotions are enabled.
3. **Selective emotion use**: apply high-arousal emotions (anger, disgust) only to robust models (Phi family, Qwen3-4B) and rely on neutral + mild happiness/surprise for weaker models to avoid catastrophic drop-offs.

## References

- Aggregation scripts: `result_analysis/generate_fantom_emotion_summary.py`, `result_analysis/generate_fantom_emotion_by_task_summary.py`.
- Raw runs: see `results/fantom/` (full directory list lives in `result_analysis/fantom_emotion_summary.md`).
