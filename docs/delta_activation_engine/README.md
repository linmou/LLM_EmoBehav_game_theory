Delta Activation Engine
Last updated: 2025-10-21

This package computes delta activation vectors by comparing baseline representations to activation-steered representations. It now contains two parallel pipelines:
- Baseline pipeline (text probes, no chat templates)
- Chat pipeline (prompts built via PromptFormat so Instruct models use their chat templates)

Layout
- delta_activation_engine/
  - config/
    - job.py — base job config (YAML → DeltaActivationJobConfig)
    - chat_job.py — chat job config (YAML → DeltaActivationChatJobConfig)
  - backends/
    - base.py — BaseBackend interface
    - hf.py — HF implementation (HFBackend, select_middle_third_layers)
  - datasets/
    - probes.py — DeltaProbesDataset (adapts list[str] → BenchmarkItem)
  - prompts/
    - wrappers.py — DeltaProbesPromptWrapper (uses PromptFormat)
    - probes_texts.py — get_generic_probes() (canonical 5 templates)
  - pipelines/
    - runner.py — run_job() baseline orchestrator
    - chat_runner.py — run_job_chat() chat-template-aware orchestrator
  - io/
    - files.py — simple save helpers (save_npz_vector, save_json)

Compatibility shims keep old imports working (config.py, runner.py, chat_runner.py, io_utils.py, probes.py, backends.py).

Pipelines
- Baseline (no chat template):
  - Uses get_generic_probes() strings directly
  - run_job(cfg, backend) computes baseline, then steered vectors per emotion × intensity
  - Saves baseline.npz, deltas/*.npz, metadata.json

- Chat (chat template aware):
  - Builds prompts through DeltaProbesDataset + DeltaProbesPromptWrapper + PromptFormat
  - Preserves model’s chat template behavior for Instruct models (Qwen, Llama, etc.)
  - Output under …/chat/<model>_<timestamp>/

Usage
- Baseline CLI (unchanged):
  - python -m delta_activation_engine.cli --config config/delta_activations/qwen2_5b_default.yaml
- Chat CLI (separate):
  - python -m delta_activation_engine.cli_chat --config config/delta_activations/chat_smoke.yaml
- Programmatic:
  - from delta_activation_engine.pipelines.runner import run_job
  - from delta_activation_engine.pipelines.chat_runner import run_job_chat

Config
- Base job (config.job.DeltaActivationJobConfig):
  - model_path, emotions, intensities, output_dir
  - loading_config, repe_eng_config
- Chat job (config.chat_job.DeltaActivationChatJobConfig):
  - Same as base plus prompt_config:
    - benchmark_name, task_type, probes (or probe_source: generic), enable_thinking

Backends
- backends.base.BaseBackend contract:
  - get_repr(prompts, steered=False, emotion=None, intensity=None) -> np.ndarray
  - get_run_metadata() -> dict
- backends.hf.HFBackend:
  - HF model path with RepE readers; last-layer last-token averaging
  - Control layers = middle third of hidden layers
  - Operator = linear_comb

Testing
- All tests under tests/delta_activation_engine/ remain green after the refactor.
  - config parsing, layer selection, probes, aggregation, intensity invariance, e2e small, metadata snapshot

Extend
- Add a new backend under backends/ and swap in via code that constructs the backend.
- Add new prompt sources in prompts/ or expand dataset adapters under datasets/.
- Keep compatibility shims temporarily if external users import old paths; remove when safe.

Notes
- Keep code small and focused: KISS, YAGNI. No hidden defaults in dataclasses.
- Prefer the middle third of layers; last-layer last-token averaged across probes.
- Avoid modifying the original CLI; chat pipeline stays separate for clean comparisons.
