# Benchmark Dataset Migration Guidance
Updated: 2025-10-01 | Commit: TBD

This guidance describes a simple, test-first, parity-focused process to migrate any new benchmark dataset into `emotion_experiment_engine`.

Principles (KISS, YNGNI)
- One dataset class, one prompt wrapper. Use `augmentation_config` to toggle modes (e.g., CoT vs zero-shot). Default to the common case (CoT if required).
- TDD in small steps: write failing tests, add minimal code to pass, then refactor. Run regression after each step.
- Parity over guesswork: verify `__getitem__` parity vs upstream repo and optional prompt text equality when requested.
- Real data first: dry-run and tests should work against the real dataset paths, not only stubs.
- No extra scripts. Always run through the series runner and `--dry-run`.

Files To Add
- Dataset: `emotion_experiment_engine/datasets/<dataset>.py`
  - Parse the real data (CSV/JSONL/etc.) → `List[BenchmarkItem]`
  - Strict evaluation logic: return deterministic scores; fail fast on invalid inputs
  - Optional: `shuffle_options_seed` kwarg. Keep one `random.Random(seed)` per dataset and shuffle `[inc1, inc2, inc3, correct]` once per row to mirror upstream behavior.
- Prompt wrapper: `emotion_experiment_engine/<dataset>_prompt_wrapper.py`
  - Exact format parity. Make CoT a mode in the one wrapper (default `gpqa_mode='cot'`) via `augmentation_config`.
  - For deterministic CoT in tests: accept `augmentation_config["<dataset>_cot_reasoning"]` or a callable provider.
- Registry: `emotion_experiment_engine/benchmark_component_registry.py`
  - Add `("<dataset>", "*")` to `BENCHMARK_SPECS` mapping to your dataset and wrapper.
- Task doc: `tasks/<dataset>_migration_plan.md` (project-specific plan)
- Doc record: `emotion_experiment_engine/claude_doc/doc_update_record/documentation_update_record_vX.Y.Z_<dataset>.md`

TDD Workflow (Red-Green-Refactor)
1) Red
   - Unit tests: dataset parsing + evaluation; `__getitem__` parity (question/context/id/options ordering); prompt wrapper validation + exact-string build.
   - Integration test: build `PromptFormat` with a real tokenizer (no network) and ensure the prompt contains expected sections; don’t assert the entire string unless required.
   - E2E parity test: import upstream loader and prompt builder; set deterministic seed; assert question/options/correct parity and (if requested) prompt string equality.
2) Green
   - Implement minimal dataset + wrapper + registry to satisfy tests.
   - Keep a single wrapper; switch modes via `augmentation_config` (CoT default if needed).
3) Refactor
   - Remove duplication; keep code short; push all knobs into `augmentation_config`.
   - Guard heavy deps for dry-run (see below).
4) Regression
   - Run the full repository tests; then run `--dry-run` with the real data path.

Parity Controls
- Options order: If the upstream repo shuffles options, add a dataset kwarg `shuffle_options_seed`. Use a single `random.Random(seed)` member on the dataset and call `.shuffle()` once per row to reproduce the upstream sequence. Re-seeding each row is wrong (identical permutation per row).
- Prompt equality: Build the wrapper to return the exact upstream text when needed (e.g., zero-shot CoT); use `augmentation_config["<dataset>_cot_reasoning"]` or a provider to supply deterministic CoT in tests.

Real-Data Dry Run (No Scripts)
- Prefer a series YAML under `config/` and run the official runner with `--dry-run`.
- Example:
  - `CUDA_VISIBLE_DEVICES=2,3 python -m emotion_experiment_engine.emotion_experiment_series_runner --config config/<dataset>_qwen3_think_series.yaml --name <RunName> --dry-run`
- Requirements for dry-run to pass without vLLM:
  - In `emotion_experiment_engine/experiment.py`, lazily import heavy modules inside GPU setup; load tokenizer via `transformers.AutoTokenizer` as fallback to avoid hard vLLM dependency.
  - In `neuro_manipulation/utils.py`, guard `from vllm import LLM` in a try/except and provide a stub `LLM` class if import fails.

Checklist (Do In Order)
- [ ] Create `tasks/<dataset>_migration_plan.md` based on this guidance.
- [ ] Add unit + integration + E2E parity tests.
- [ ] Implement dataset + wrapper + registry (single wrapper; CoT via augmentation; CoT default if required).
- [ ] Ensure dataset supports optional `shuffle_options_seed` parity.
- [ ] Update docs: doc update record; (optionally) README note.
- [ ] Run `--dry-run` with the real dataset path; fix any import/path issues.
- [ ] Run full series without `--dry-run` when ready.

Common Pitfalls (and Fixes)
- “No module named vllm” on dry-run: lazy-import heavy deps; guard vLLM import with a stub.
- Prompt mismatch: implement a single wrapper returning exact upstream text; use `augmentation_config` to switch modes (don’t add a second class).
- Option order mismatch: pass a seed and keep one RNG per dataset (don’t re-seed per row).
- Unnecessary scripts: do not add; rely on runner + `--dry-run`.

Notes From Prior Feedback
- Remove disposable scripts; rely on the series runner CLI.
- Use real datasets in dry-run.
- E2E parity must include prompt-string equality where requested.
- Implement CoT as an augmentation method and set it as default when appropriate.

