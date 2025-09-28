# SWE-bench Migration Plan


## Purpose
Establish a repeatable pathway to run SWE-bench evaluations (the SWE repo is at: /data/home/jjl7137/SWE-bench) under the RepE/vLLM emotion control infrastructure so we can compare neutral versus emotion-activated models using the official SWE-bench harness.

## Overview
This migration connects the existing `emotion_experiment_engine` (inference, RepE control, batching) with the SWE-bench repository (dataset access and Docker-based evaluation). The end state is a scripted workflow that, for each emotion/intensity, generates patches with `rep-control-vllm` and immediately scores them with the SWE-bench harness.

## Retrieval Strategy
- Mode: Offline retrieval (precomputed) only. Online RAG (live retrieval during inference) is not supported.
- Approach: Use SWE-bench scripts to generate BM25 retrieval hits and materialize a text dataset with `text_inputs`. Our runtime reads this dataset; it does not build indices, clone repos, or run BM25 during inference.
- Storage: Place all retrieval outputs and generated datasets under `./cache`.

## Current State
- Emotion experiment engine supports many benchmarks through dataset/prompt registries but has no SWE-bench adapter.
- RepE control pipeline already loads models with vLLM and supports emotion intensities.
- SWE-bench repo provides dataset utilities, prompting helpers, and the authoritative harness (`python -m swebench.harness.run_evaluation`).
- No glue exists between the two systems; SWE-bench prompting is only available via standalone scripts.

## Target Architecture
1. **SWEbenchDataset** (`emotion_experiment_engine/datasets/swebench.py`)
   - Load a precomputed SWE-bench text dataset (with `text_inputs`) produced by SWE-bench offline scripts.
   - Supply `text_inputs` to the model and capture responses (predictions JSONL emission handled in Phase 2).
2. **Registry Wiring**
   - Map `("swebench", "patch")` to the new dataset (prompt wrapper optional for offline parity) in `benchmark_component_registry.py`.
3. **Series Runner Integration**
   - Use the existing series runner entrypoint: `python -m emotion_experiment_engine.emotion_experiment_series_runner --config <config.yaml>`.
   - Provide a SWE-bench config profile that iterates over emotion × intensity, generates predictions, and then launches the SWE-bench harness on each predictions file.
   - Aggregate pass/fail metrics into the experiment’s result tables.
 

## Migration Steps

### Phase 0 – Prerequisites
- [x] Confirm both repos are on the `llm_fresh` environment and share HF caches.
- [x] Document GPU/CPU availability and set default vLLM loading config (`max_model_len`, `tensor_parallel`).
- [x] Precompute retrieval results and text dataset (offline) and store under `./cache`:
  - Retrieval hits (BM25):
    - `python -m swebench.inference.make_datasets.bm25_retrieval \\
      --dataset_name_or_path SWE-bench/SWE-bench_Lite \\
      --document_encoding_style file_name_and_contents \\
      --output_dir ./cache/retrieval_results`
  - Materialize text dataset (default prompt style-3, BM25 source):
    - `python -m swebench.inference.make_datasets.create_text_dataset \\
      --dataset_name_or_path SWE-bench/SWE-bench_Lite \\
      --output_dir ./cache/datasets \\
      --prompt_style style-3 --file_source bm25 \\swebench_data_prep
      --retrieval_file ./cache/retrieval_results/<name>.retrieval.jsonl \\
      --k 20 --max_context_len 32768 --tokenizer_name llama`
- [x] Provide helper script `python -m emotion_experiment_engine.swebench_data_prep` to orchestrate the above commands with cache management and dry-run support.

### Phase 1 – Minimal Adapter
- [x] Create `SWEbenchDataset` that streams the Lite split and passes through `text_inputs` (predictions JSONL handled in Phase 2).
- [x] Register the new benchmark in `benchmark_component_registry.py`.
- [x] Add unit tests to `emotion_experiment_engine/tests` covering dataset length and collate contract, parity vs HF dataset, and registry lookup.

Dry‑run sanity check (dataset wiring)
- After wiring the dataset + wrapper and creating a minimal SWE-bench config, validate that datasets are discovered and load without executing generation:
  - `python -m emotion_experiment_engine.emotion_experiment_series_runner --config <swebench.yaml> --dry-run`
  - Confirms `./cache/datasets/<name>` is readable and registry mapping (`"swebench","patch"`) resolves.

TDD (Phase 1)
- Prompt Wrapper
  - Red: failing unit test asserting default `style-3` selection and idempotent passthrough for offline `text_inputs`.
  - Green: minimal wrapper honoring `style-3` and no-op on precomputed prompts.
  - Parity: assert wrapper output equals `text_inputs` from SWE-bench text dataset for N=10 sampled instances.
  - Regression: run full test suite and mypy on touched modules.
- Dataset
  - Red: failing test for dataset length, sample item keys (`instance_id`, `text_inputs`), and collate returning batch of strings.
  - Green: implement dataset loader for HF `save_to_disk` datasets under `./cache/datasets`.
  - Parity: compare `text_inputs` string equality between our loader and directly reading from the HF dataset object for N=10 instances.
  - Regression: run suite; validate predictions JSONL schema contains `instance_id` + `model_patch`.
- Registry
  - Red: failing test for registry lookup `("swebench","patch")` returning dataset + wrapper classes.
  - Green: wire registry mapping.
  - Regression: suite green.

### Phase 2 – Orchestra Generation
- [x] Add a SWE-bench config (`config/swebench_series_lite.yaml`) for `emotion_experiment_engine.emotion_experiment_series_runner` that performs generation-only runs against cached datasets.
- [x] Extend `EmotionExperiment._post_process_batch` to tag each `ResultRecord` with predictions path/run_id for downstream reporting and persist JSONL predictions.
- [x] Produce a smoke test run (≤5 instances) to validate the end-to-end generation flow. (Dry run succeeds with `text_inputs` dataset at `./cache/datasets/SWE-bench_Lite_text_inputs_dataset`.)
- Note: Mirrors the “Deferred Evaluation Workflow” in `emotion_experiment_engine/README.md` — generate first, evaluate later.

TDD (Phase 2)
- Series Runner Path (Generation)
  - Red: failing test invoking `python -m emotion_experiment_engine.emotion_experiment_series_runner --config <swebench.yaml>` on a 3–5 instance subset; assert predictions JSONL file created with correct schema; assert no harness side-effects.
  - Green: implement SWE-bench config handling in the series runner to write predictions only.
  - Regression: full suite green.
- Post-process Tagging
  - Red: failing unit test expecting `ResultRecord` to include `predictions_path` and `run_id`.
  - Green: add fields during post-processing. (Implemented via `emotion_experiment_engine/tests/test_swebench_predictions.py`.)
  - Regression: suite green.

### Phase 3 – Equivalent Evaluation & Reporting 
- [ ] Evaluate generated predictions via SWE-bench harness (deferred step):
  - For each predictions JSONL from Phase 2, run `python -m swebench.harness.run_evaluation` and store logs/reports.
- [ ] Merge harness report JSON into experiment summaries (resolved counts, pass@1).
- [ ] Create a result manifest at `results/swebench/<model>/<timestamp>.json` storing emotion, intensity, predictions path, harness run ID, and pass rate.
- [ ] Hook the evaluation + merge into an optional follow-up step or helper script consistent with the deferred workflow.

TDD (Phase 3)
- Harness Evaluation
  - Red: failing integration test that runs the harness on a small predictions file; asserts reports are created and contain per-instance status.
  - Green: wire harness invocation and artifact collection.
  - Parity: verify harness accepts our predictions JSONL without transformation and produces expected log structure.
  - Regression: suite green.
- Reporting Merge
  - Red: failing test when merged summary lacks `pass@1`, `resolved_count`.
  - Green: implement merge; compute pass rate from harness reports.
  - Parity: for the smoke subset, assert our merged metrics equal values computed directly from harness logs.
  - Regression: suite green.
- Result Manifest
  - Red: failing test for manifest schema and file path under `results/swebench/<model>/`.
  - Green: write manifest; validate round-trip load.
  - Regression: suite green.

 

## Recommended Defaults (SWE-bench-aligned)
- Dataset: `SWE-bench_Lite` (start here; scale later)
- Prompt style: `style-3`
- Retrieval: BM25 (offline), top-k files `k=20`
- Token gating: `max_context_len=32768` with `tokenizer_name=llama`
- Predictions format: JSONL with `instance_id` and `model_patch` keys (harness expects `model_patch`)

Parity Criteria
- Prompt Parity: `text_inputs` loaded by our dataset must exactly match strings in the SWE-bench text dataset for sampled instances.
- Schema Parity: predictions JSONL strictly contains `instance_id` and `model_patch` keys and is accepted by `swebench.harness.run_evaluation` without rewrite.
- Metric Parity: merged pass@1 and resolved counts equal values recomputed from harness logs for the same run.

## Key Risks & Mitigations
- **Invalid patches**: Add validation in the dataset to drop empty/non-diff outputs before the harness run.
- **Harness throughput**: Start with small subsets; enqueue full runs overnight.
- **vLLM/GPU memory**: Provide conservative defaults; allow override in config.
- **Repo cloning cost** (if retrieval enabled): Cache clones per repo/commit; limit instances per batch.

## Dependencies
- Existing RepE configs (`get_repe_eng_config`).
- SWE-bench dataset cache (`hf_cache`) and Docker access for the harness.
- Working `neuro_manipulation` package (prompt formats, pipeline loader).
- For offline retrieval generation only: Java + Pyserini required when running SWE-bench `bm25_retrieval.py` (not needed at runtime in this repo).

## Definition of Done
- A single command produces emotion-conditioned predictions and harness evaluation for at least two emotions on SWE-bench Lite.
- Results include per-emotion pass rates and link back to predictions/harness logs.
- Repository documentation references the new workflow with sample command-line usage.

## Cache Layout
- `./cache/retrieval_results/`: BM25 retrieval outputs (`*.retrieval.jsonl`).
- `./cache/datasets/`: Materialized text datasets with `text_inputs` (by split).
