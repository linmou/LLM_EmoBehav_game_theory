# GPQA Migration Plan

Updated: 2025-09-30 · commit: TBD

## Purpose
Integrate GPQA (main/extended/diamond) as a first-class benchmark in `emotion_experiment_engine` so we can run emotion-conditioned evaluations on graduate-level multiple-choice questions.

## Overview
GPQA is single-answer multiple-choice. We treat it as MC1 and reuse the existing TruthfulQA MC1 prompt formatting to avoid reinventing formatting logic. Evaluation is strict, case-insensitive exact-text match.

## Data Sources
- Local zip: `/data/home/jjl7137/gpqa/dataset.zip` (password: `deserted-untie-orchid`)
  - Files inside: `dataset/gpqa_main.csv`, `dataset/gpqa_extended.csv`, `dataset/gpqa_diamond.csv`
- Optional: You can point `BenchmarkConfig.data_path` directly to any CSV subset you prepare.

Extraction example:
```bash
unzip -P deserted-untie-orchid /data/home/jjl7137/gpqa/dataset.zip -d /data/home/jjl7137/gpqa/
# CSVs at: /data/home/jjl7137/gpqa/dataset/gpqa_*.csv
```

## Target Architecture
1. Dataset: `emotion_experiment_engine/datasets/gpqa.py` → `GPQADataset`
   - CSV → `BenchmarkItem` list
   - `metadata["options"] = [correct, wrong1, wrong2, wrong3]`
   - `evaluate_response`: 1.0 on exact case-insensitive match, else 0.0
2. Prompt wrapper: `emotion_experiment_engine/gpqa_prompt_wrapper.py`
   - Single class `GPQAPromptWrapper` with augmentation-controlled modes
   - Default mode: CoT (`gpqa_mode='cot'`), matches zero_shot_chain_of_thought_prompt structure
   - Optional mode: plain zero-shot (`gpqa_mode='zero_shot'`)
3. Registry: `emotion_experiment_engine/benchmark_component_registry.py`
   - Map `("gpqa", "*")` → (dataset=GPQADataset, wrapper=GPQAPromptWrapper)

4. Parity Controls
   - Dataset arg `shuffle_options_seed` reproduces upstream GPQA’s choice-order shuffling:
     - Baseline builds `[Incorrect1, Incorrect2, Incorrect3, Correct]`, then calls `random.shuffle` with a fixed seed set once.
     - We mirror this with one `random.Random(seed)` kept on the dataset instance and shuffle per row in load order.

## TDD Plan (Red-Green-Refactor)
1. Red
   - Add tests: `emotion_experiment_engine/tests/unit/datasets/test_gpqa_dataset.py`
     - Factory creates dataset via registry
     - Items parsed from minimal CSV
     - Eval returns 1.0 for correct/0.0 for incorrect
2. Green
   - Implement `GPQADataset`, `GPQAPromptWrapper`, and registry entry
3. Refactor
   - Keep code minimal; no overdesign. Update docs.

Status
- [x] Tests added (Red)
- [x] Minimal implementation (Green)
- [x] Registry wiring
- [x] Docs updated (README + doc record)

Note: Full repo test run surfaces unrelated Python 3.9 union-type (`|`) issues in other tests during collection; not introduced by this change.

## Usage Examples
Python (direct factory usage):
```python
from pathlib import Path
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.benchmark_component_registry import create_benchmark_components

config = BenchmarkConfig(
    name="gpqa",
    task_type="main",  # or "extended", "diamond"
    data_path=Path("/data/home/jjl7137/gpqa/dataset/gpqa_main.csv"),
    base_data_dir="/data/home/jjl7137/gpqa/dataset",
    sample_limit=None,
    augmentation_config=None,
    enable_auto_truncation=False,
    truncation_strategy="right",
    preserve_ratio=1.0,
    llm_eval_config=None,
)

prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
    benchmark_name="gpqa", task_type=config.task_type, config=config, prompt_format=None,
    shuffle_options_seed=1234,  # parity: match GPQA repo choice order
    augmentation_config={
        "gpqa_mode": "cot",  # default; set "zero_shot" to disable CoT
        "gpqa_cot_reasoning": "Because of X, Y, Z, the best answer is A.",
        # or provide a callable provider to generate CoT dynamically
        # "gpqa_cot_provider": lambda question, options: "...",
    },
)

item = dataset[0]
score = dataset.evaluate_response("some answer", item["ground_truth"], config.task_type, item["prompt"])
```

Series runner (concept): Add a `benchmark` block with `name: gpqa` and appropriate `task_type` in your YAML, then run `python -m emotion_experiment_engine.emotion_experiment_series_runner --config <config.yaml>`.

## Acceptance Criteria
- Dataset loads from CSV and yields items with options in metadata
- Evaluation: 1.0 for exact case-insensitive match; 0.0 otherwise
- Registry builds prompt wrapper + dataset for `("gpqa", any)`
- Docs updated and tests passing for the new module
- Prompt text equals GPQA zero-shot baseline format when using GPQAZeroShotPromptWrapper
  and option order parity is enabled via `shuffle_options_seed`

## Follow-ups (Optional)
- Add helper to auto-extract from the password-protected zip to a default data dir
- Add HF fallback loader for `idavidrein/gpqa` when network is permitted
- Provide a small curated subset CSV for quick local smoke tests
