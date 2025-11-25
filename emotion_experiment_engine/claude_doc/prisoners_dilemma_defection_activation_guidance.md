# Prisoner's Dilemma Defection Activation Guidance
# Updated: 2025-11-25

Goal: Extract a defection activation direction (cooperate as negative) from `data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json` using RepReaders.

## Prompt template
Use a single user turn so the last token is the assistant’s letter:
```
User: Imagine you are Alice and consider the following scenario: {description}
Choices:
A) {opt_a}
B) {opt_b}
Assistant: {label}
```
- `rep_token = -1` targets the `{label}` token.
- Build the whole string as one user message via `PromptFormat`; keep `max_length≈256` by truncating description only.

## Option assignment and labels
- Per scenario, randomly decide the ordering:
  - Case 1: A=defect, B=cooperate
  - Case 2: A=cooperate, B=defect
- Build a contrastive pair with identical user text and option order; only the assistant label flips:
  - Positive (defection): `Assistant: A` if A=defect else `Assistant: B`
  - Negative (cooperation): the opposite letter
- Record the option map to decode labels during validation.

## Extraction settings (RepReader)
- Layers: middle third of model layers (same rule as `select_middle_third_layers`).
- `n_difference = 1`, `direction_method = pca`, `rep_token = -1`.
- Tokenizer/model from the target run; `max_length≈256`.
- Cache outputs under `neuro_manipulation/representation_storage/prisoners_dilemma_defect_<hash>.pkl` with metadata (dataset hash, option maps, model, layers, prompt settings).

## Validation
- Split 50/50 train/holdout at the scenario level.
- On holdout, use the benchmark registry entry in `emotion_experiment_engine/benchmark_component_registry.py`:
  - Key: `("game_theory", "*")`
  - `BenchmarkSpec(dataset_class=GameTheoryDataset, answer_wrapper_class=IdentityAnswerWrapper, prompt_wrapper_class=GameBenchmarkPromptWrapper)`
- Config reference: `config/new_game_theory_config.yaml` (reuse loading/formatting knobs).
- Metric: per-layer accuracy that projection(defect) > projection(cooperate) using learned signs. Flag weak layers.

## Run sketch
1) Load JSON → build paired prompts with randomized A/B order and labels.  
2) Split 50/50.  
3) Train RepReader on train pairs with settings above.  
4) Validate on holdout; report per-layer accuracy.  
5) Persist vectors + metadata; note option maps for reproducibility.
