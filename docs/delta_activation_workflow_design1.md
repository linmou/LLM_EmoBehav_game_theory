# Delta Activation Workflow Plan

Last updated: 2025-10-13

This document specifies the YAML-driven workflow to compute delta_activation vectors by comparing activation-steered (emotion-activated) runs to baseline (no steering), aligned with:
- emotion_experiment_engine/experiment.py for config loading, tokenizer parity, and RepE reader loading
- /home/jjl7137/delta_activations/delta_activations.py for representation extraction pattern (last-layer, last-token; averaged over 5 generic instruction probes)

KISS principles apply: do the simplest thing that works, keep the API stable so we can swap in a faster vLLM backend later.

---

## Scope

- Compute per-emotion, per-intensity delta vectors: `delta = steered_activation − baseline_activation`.
- Representation: last-layer, last-token hidden state averaged over a fixed set of 5 generic probes (exactly as in delta_activations.py).
- Control layers: the middle third of layers (computed dynamically per model).
- Controller/operator: `linear_comb` (default), `token_pos=None`, `normalize=False`.
- Backends:
  - v1: HF-only, using WrappedReadingVecModel to inject activations.
  - v2 (future): vLLM path via RepControlVLLMHook; only plan now—do not implement if hidden states are not trivially available.
- Outputs: `.npz` (vectors) + `.json` (metadata), written under a root output dir from YAML.

---

## File Architecture (minimal)

```
delta_activation_engine/
  runner.py                # CLI entry: reads YAML, orchestrates workflow
  config.py                # Dataclasses for YAML→config; no defaults in dataclasses
  hf_backend.py            # Steered-baseline extraction via HF + WrappedReadingVecModel
  vllm_backend.py          # (stub) Placeholder for speed path with RepControlVLLMHook
  probes.py                # 5 generic instruction prompts (verbatim from delta_activations.py)
  io_utils.py              # Save/load helpers for npz+json; probe hashing, metadata

config/
  delta_activations/
    qwen2_5b_default.yaml  # Example: model_path, emotions, intensities, output_dir, repe_eng_config, loading_config

docs/
  delta_activation_workflow.md   # This plan

tests/
  delta_activation_engine/
    test_config_parsing.py
    test_layer_selection.py
    test_probes.py
    test_aggregation.py
    test_intensity_invariance.py
    test_end_to_end_small.py
    test_parity_zero_activation.py
```

Rationale: keep responsibilities tight; avoid over-configuration; prefer small modules with clear edges. Place new code under `delta_activation_engine/` (not `result_analysis/`).

---

## YAML Config Shape

Aligned with emotion_experiment_engine data models; reuse its loading config shape and RepE config. Example:

```yaml
# config/delta_activations/qwen2_5b_default.yaml
model_path: /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct
emotions: [anger, happiness, sadness, disgust, fear, surprise]
intensities: [0.5, 1.0, 1.5]
output_dir: results/delta_activations

loading_config:
  model_path: /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct
  gpu_memory_utilization: 0.85
  tensor_parallel_size: null
  max_model_len: 4096
  enforce_eager: true
  quantization: null
  trust_remote_code: true
  dtype: float16
  seed: 0
  disable_custom_all_reduce: false
  additional_vllm_kwargs: {}

repe_eng_config:
  control_method: reading_vec
  block_name: decoder_block
  rep_token: "<REP>"
  data_dir: data/stimulus/text/
  n_difference: 128
  direction_method: mean-diff
  emotions: [anger, happiness, sadness, disgust, fear, surprise]
```

Flags kept minimal; YAML is the single source of truth.

---

## CLI Usage

```
conda activate llm_fresh
python -m delta_activation_engine.cli --config config/delta_activations/qwen2_5b_default.yaml
```

This runs the HF backend and writes outputs under the YAML's `output_dir`.

---

## Chat-Template Pipeline (New)

Goal: observe the impact of model chat templates on delta activations without changing the existing CLI.

- Entry: `delta_activation_engine/chat_runner.py` (programmatic) and optional module `delta_activation_engine.cli_chat` (if needed later).
- Config: `delta_activation_engine/chat_config.py` defines a strict YAML schema that extends the base job with `prompt_config`:
  - `benchmark_name` (e.g., `delta_probes`), `task_type` (e.g., `default`)
  - `probes` (explicit list) or `probe_source: generic` (uses built-ins)
  - `enable_thinking` (optional; passed to wrappers)
- Dataset: `delta_activation_engine/datasets.DeltaProbesDataset` adapts probe strings into `BaseBenchmarkDataset` items so wrappers can build prompts via `PromptFormat.apply_chat_template`.
- Wrappers: Reuse `emotion_experiment_engine` prompt wrappers. We try the component registry first; if the `(benchmark_name, task_type)` pair is unregistered, we fall back to `get_benchmark_prompt_wrapper` + our dataset.
- Outputs: Saved under `…/chat/<model>_<timestamp>/` with `metadata.json` including `chat_template` and a `prompt_config` snapshot.

Example minimal YAML for chat pipeline:

```
model_path: /models/DUMMY
emotions: [anger]
intensities: [0.0, 1.0]
output_dir: results/delta_activations
loading_config: { model_path: /models/DUMMY, max_model_len: 4096 }
repe_eng_config: { control_method: reading_vec, block_name: decoder_block, rep_token: "<REP>", data_dir: data/stimulus/text/, n_difference: 8, direction_method: mean-diff, emotions: [anger] }
prompt_config:
  benchmark_name: delta_probes
  task_type: default
  probes: ["Say hello", "Summarize: test"]
```

Programmatic run (example):

```
from delta_activation_engine.chat_config import load_chat_job_config_from_yaml
from delta_activation_engine.chat_runner import run_job_chat

cfg = load_chat_job_config_from_yaml("/path/to/chat_job.yaml")
out_dir = run_job_chat(cfg)  # writes npz + json under results/delta_activations/chat/
```

Notes:
- We do not touch `delta_activation_engine/cli.py` per request; this is a parallel pipeline to compare with the baseline.
- Backends: The chat runner accepts an injected backend for tests; production uses the existing `HFBackend`.

---

## Class Diagram (ASCII)

```
+--------------------------+         uses          +----------------------+
| DeltaActivationRunner    |---------------------->|  HFSteerBackend      |
|  - cfg: JobConfig        |                      |  + get_repr(...)     |
|  + run()                 |                      +----------------------+
|  + _save(delta, meta)    |                      ^
+------------+-------------+                      |
             | uses                                |
             v                                     |
+--------------------------+         provides      |
| DeltaActivationJobConfig |<----------------------+
|  - model_path            |       +----------------------------+
|  - emotions: List[str]   |       |  probes.py (ProbeSet)      |
|  - intensities: List[float]      |  + get_generic_probes()    |
|  - output_dir            |       +----------------------------+
|  - loading_config        |
|  - repe_eng_config       |
+--------------------------+
```

Key points:
- `HFSteerBackend.get_repr(prompts, steered=False, emotion=None, intensity=None)` returns a single vector: last-layer last-token representation averaged over prompts.
- `DeltaActivationRunner` calls baseline once, then loops over emotions×intensities, computes deltas, and saves artifacts.

---

## Dataflow

```
YAML → JobConfig (reusing emotion_experiment_engine config structures)
   → load_tokenizer_only(model_path)            # reuse path via emotion_experiment_engine
   → get_repe_eng_config(model_path, yaml)      # reuse RepE config builder
   → setup_model_and_tokenizer(from_vllm=False) # HF model for reader building (experiment.py pattern)
   → detect num_hidden_layers → pick middle third as control layers (ModelLayerDetector)
   → load_emotion_readers(repe_eng_config, model(HF), tokenizer, hidden_layers)
   → build 5 generic probes (probes.py)

Baseline pass (HF):
   → tokenize(probes, pad=eos, trunc=256)
   → model(**inputs), collect hidden_states[-1][:, -1, :]
   → average across probes → base_activation

For each (emotion, intensity):
   → wrap HF model with WrappedReadingVecModel
   → wrap_decoder_block for control layers
   → set_controller(activations=emotion_vecs, operator=linear_comb, token_pos=None, normalize=False)
   → forward same probes, collect hidden_states[-1][:, -1, :]
   → average → steered_activation
   → delta = steered_activation − base_activation
   → save npz + json (with metadata: model_path, layers, intensity, operator, probe_hash)
```

Memory hygiene:
- Use `torch.no_grad()`, small `max_length=256`.
- Clear CUDA cache between big steps if needed.

---

## Output Layout

```
{output_dir}/
  {model_basename}_{timestamp}/
    metadata.json                 # model_path, control_layers, probe_hash, config snapshot
    baseline.npz                  # vector
    deltas/
      emotion=anger_int=0.5.npz   # vector
      emotion=anger_int=1.0.npz
      ...
```

All `.npz` store a single array: `vector`.

---

## Test Plan (hierarchical)

### Unit Tests
- config parsing (tests/delta_activation_engine/test_config_parsing.py)
  - Validate required fields; error on missing model_path/output_dir/emotions/intensities.
- probe set (test_probes.py)
  - Exactly 5 templates, stable content; hash matches expected.
- layer selection (test_layer_selection.py)
  - Given num_layers ∈ {12, 24, 32}, computed middle third indices match spec.
- aggregation (test_aggregation.py)
  - Given synthetic hidden_states, last-layer last-token averaging returns correct shape and values.
- intensity invariance (test_intensity_invariance.py)
  - With intensity=0 (or zero vector), delta norm ≈ 0.
  - Doubling intensity increases ||delta|| monotonically on a mock model.
- output writing (test_aggregation.py or separate)
  - Files created in `output_dir` with expected naming and metadata keys.

Notes:
- Each test file begins with a comment: which module it covers and purpose.
- No defaults in dataclasses; tests ensure explicitness.

### Integration Tests (small, CPU-friendly)
- end-to-end small (test_end_to_end_small.py)
  - Tiny HF model (or mocked HF model) + mocked RepReader outputs (small vectors).
  - Run 2 emotions × 2 intensities; assert `.npz` files exist; shapes match hidden_dim; metadata sane.
- parity: zero-activation (test_parity_zero_activation.py)
  - Baseline vs. steered with zero vectors produce deltas ≈ 0.

### Parity Tests (future-enable when vLLM path exists)
- HF vs. vLLM backend parity (skipped by default for now)
  - Same config, seeds, probes; compare cosine similarity of deltas ≥ 0.99.
  - Enabled once we can reliably capture hidden states with RepControlVLLMHook.

### Regression Guard
- After each change, run the full test suite.
- mypy pass on new modules (CI step recommendation).

---

## Minimal Implementation Notes
- Reuse emotion_experiment_engine patterns and utilities end-to-end:
  - `load_tokenizer_only` for tokenizer parity
  - `get_repe_eng_config` to build RepE config
  - `setup_model_and_tokenizer(from_vllm=False)` to obtain a temporary HF model for reader extraction
  - `load_emotion_readers` for RepE vectors
- Strictly follow last-layer, last-token averaging and the 5-template probe set to mirror delta_activations.py.
- Compute control layers once from the model; pick middle third.
- vLLM/RepControlVLLMHook path is stubbed until hidden states are accessible—leave for future discussion.
