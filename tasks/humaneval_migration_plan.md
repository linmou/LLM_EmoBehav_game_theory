# HumanEval Migration Plan

Updated: 2025-10-01 | Commit: TBD

## 1) Goal
Add HumanEval as a first-class benchmark with exact prompt parity and strict evaluation via upstream `human_eval.execution.check_correctness`. Keep it KISS: one dataset class, one prompt wrapper, single registry entry.

## 2) Scope & Assumptions
- Source repo: `/home/jjl7137/human-eval`
- Data file: `/home/jjl7137/human-eval/data/HumanEval.jsonl.gz`
- Upstream APIs used: `human_eval.data.read_problems`, `human_eval.execution.check_correctness`
- Single task type; use registry wildcard `("humaneval", "*")`
- Default prompt mode: raw upstream prompt (no chat framing) to preserve parity
- No option shuffling; HumanEval is code completion, not MCQ

## 3) Files To Add
- Dataset: `emotion_experiment_engine/datasets/humaneval.py`
- Prompt wrapper: `emotion_experiment_engine/humaneval_prompt_wrapper.py`
- Registry entry: `emotion_experiment_engine/benchmark_component_registry.py`
  - `( "humaneval", "*" ) -> BenchmarkSpec(dataset=HumanEvalDataset, answer=IdentityAnswerWrapper, prompt=HumanEvalPromptWrapper)`
- Tests: `tests/benchmarks/humaneval/` (unit + integration + parity)
- Optional config: `config/humaneval_qwen_series.yaml` (for dry-run/smoke)

## 4) Dataset Design (HumanEvalDataset)
- Loader
  - Read gzipped jsonl exactly like upstream (`gzip.open`): `/home/jjl7137/human-eval/data/HumanEval.jsonl.gz`
  - Map each row to `BenchmarkItem`:
    - `id`: `task_id`
    - `input_text`: `prompt` (the code stub with signature/docstring and a trailing `pass`)
    - `context`: `None`
    - `ground_truth`: full upstream task dict (must include `test`, `entry_point`, `canonical_solution`)
    - `metadata`: `{ "entry_point": <entry_point>, "source": "humaneval" }`
  - Respect `sample_limit` from `BenchmarkConfig`
- __getitem__
  - Defer prompt building to wrapper; call with `question=item.input_text`, `context=""`, `options=None`
- Evaluation
  - `evaluate_response(response, ground_truth, task_name, prompt)` calls
    `human_eval.execution.check_correctness(ground_truth, completion=response, timeout=self.eval_timeout, completion_id=None)`
  - Return `1.0` if `passed` else `0.0`; store failure detail string to `self._last_eval_errors` for debugging
  - Config knobs: `eval_timeout: float` (default 3.0), `eval_workers: int` (used only if we later batch via threads)
- Metrics
  - `get_task_metrics(task_name) -> ["accuracy"]`
  - `compute_split_metrics(records)` → `{ "pass_rate": passed / total }` (pass@1)
- Parity
  - Exact prompt equality; no augmentation by default

## 5) Prompt Wrapper (HumanEvalPromptWrapper)
- Default mode: raw pass-through (return the exact `question` string)
- Accept signature args for consistency: `user_messages`, `emotion`, `enable_thinking`, `augmentation_config` (ignored by default)
- Optional mode: if `augmentation_config.get("humaneval_mode") == "chat"`, build a minimal chat prompt via `PromptFormat`; OFF by default

## 6) Registry Update
Add to `emotion_experiment_engine/benchmark_component_registry.py`:
```
("humaneval", "*") : BenchmarkSpec(
    dataset_class=HumanEvalDataset,
    answer_wrapper_class=IdentityAnswerWrapper,
    prompt_wrapper_class=HumanEvalPromptWrapper,
),
```

## 7) TDD Plan (Red-Green-Refactor)
1. Red – Unit tests
   - `tests/benchmarks/humaneval/test_loader.py`
     - Dataset loads from gz; len > 0
     - `__getitem__`: prompt equals upstream `problems[task_id]["prompt"]`; ids and `metadata.entry_point` match; `ground_truth` is the full dict
   - `tests/benchmarks/humaneval/test_wrapper.py`
     - Default wrapper returns the input prompt exactly (no framing)
     - Chat mode only when explicitly enabled; default is raw
   - `tests/benchmarks/humaneval/test_eval.py`
     - For a sampled problem, evaluating the upstream `canonical_solution` returns `1.0`
     - A known-bad completion returns `0.0`
     - `compute_split_metrics` returns expected pass_rate
2. Red – Integration/Parity
   - `tests/benchmarks/humaneval/test_registry_factory.py`
     - `create_benchmark_components("humaneval", "*", ...)` wires classes correctly
   - `tests/benchmarks/humaneval/test_example_files.py`
     - Use `/home/jjl7137/human-eval/data/example_problem.jsonl` and `/home/jjl7137/human-eval/data/example_samples.jsonl`
     - Confirm 0.5 pass@1 on examples (3/6 pass)
3. Green – Minimal impl
   - Implement dataset, wrapper, registry entry to satisfy tests; keep code short
4. Refactor – Only after green
   - Remove duplication; keep default mode raw; push optional behaviors to `augmentation_config`
   - Run mypy on modified files
5. Regression (回归测试)
   - Run full repo tests; then dry-run with a real path and small `sample_limit`

## 8) Dry-Run & Smoke
- YAML sketch `config/humaneval_qwen_series.yaml`:
```
model_path: /data/home/.../Qwen/Qwen2.5-0.5B-Instruct
emotions: ["neutral"]
intensities: [0.0]
benchmark:
  name: humaneval
  task_type: main
  data_path: /home/jjl7137/human-eval/data/HumanEval.jsonl.gz
  sample_limit: 5
  enable_auto_truncation: false
  truncation_strategy: right
  preserve_ratio: 1.0
batch_size: 1
```
- Command (dry-run):
```
python -m emotion_experiment_engine.emotion_experiment_series_runner \
  --config config/humaneval_qwen_series.yaml \
  --name humaneval_smoke --dry-run
```

## 9) Risks & Mitigations
- Untrusted code execution: rely on upstream `check_correctness` sandbox, keep `timeout` low (3s)
- Prompt drift: default wrapper is raw; no chat framing unless explicitly requested
- Heavy eval time: start with small `sample_limit` and single pass@1

## 10) Checklist
- [ ] Create dataset and wrapper
- [ ] Add registry mapping `("humaneval", "*")`
- [ ] Add unit, integration, parity tests
- [ ] Dry-run with real data path; fix import/path issues
- [ ] Full test suite green (Regression)
- [ ] Optional: smoke generation run without `--dry-run`

## 11) References
- Guidance: `tasks/dataset_migration_guidance.md`
- Upstream repo: `/home/jjl7137/human-eval`
- Registry reference: `emotion_experiment_engine/benchmark_component_registry.py`
