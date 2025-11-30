# run_job_chat (`delta_activation_engine/pipelines/chat_runner.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Chat-template-aware orchestrator that renders probes through the model’s chat template before computing delta activations. Mirrors the baseline pipeline but builds prompts via PromptFormat and dataset wrappers.

## Implementation Walkthrough
1) Load tokenizer only and wrap it in `PromptFormat` to access the chat template.
2) Build `BenchmarkConfig` from `prompt_config` (benchmark name, task type, truncation disabled).
3) Choose probes: explicit `prompt_config.probes` takes precedence; otherwise `probe_source=generic` or default falls back to `get_generic_probes()`.
4) Create `DeltaProbesPromptWrapper` (system prompt empty, optional thinking flag) and `DeltaProbesDataset` to adapt probes into prompts; `_collect_prompts` pulls rendered prompts via dataset `__getitem__`.
5) Initialize backend if not provided by wrapping `DeltaActivationJobConfig` (shim) and constructing `HFBackend`.
6) Write outputs under `<output_dir>/chat/<model>_<timestamp>/`, then repeat the baseline + delta loop identical to `run_job`.
7) Record chat-specific metadata (chat template, prompt_config snapshot) alongside backend/job config in `metadata.json`.

## Key Logic
- Prompt rendering flows through the same code paths used by emotion experiments (`PromptFormat` + dataset wrapper) to ensure chat template parity.
- Probe hash computed after rendering to capture chat template effects.
- Backend injection allows lightweight testing with fake backends/tokenizers.

## Dependencies
- `neuro_manipulation.utils.load_tokenizer_only` and `neuro_manipulation.prompt_formats.PromptFormat`.
- Dataset and wrapper from `delta_activation_engine.datasets.probes` and `prompts.wrappers`.
- IO helpers and `HFBackend` as in the baseline pipeline.

## Potential Issues / Gaps
- Falls back to generic probes if `probe_source` is unrecognized; no explicit error.
- `enable_thinking` flag simply passes through to PromptFormat; no validation that the model supports thinking mode.
- Uses same hard-coded max length (256) and layer selection as baseline via HFBackend.

## Usage Example
```python
from delta_activation_engine.config import load_chat_job_config_from_yaml
from delta_activation_engine.pipelines.chat_runner import run_job_chat

cfg = load_chat_job_config_from_yaml("config/delta_activations/chat_smoke.yaml")
out_dir = run_job_chat(cfg)  # backend optional; defaults to HFBackend
print("Chat artifacts in", out_dir)
```
