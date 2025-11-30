# Delta Activation Engine Overview
Last updated: 2024-03-19 (working copy)

The delta activation engine measures how steering a model toward an emotion shifts its internal representations. It runs two parallel pipelines—baseline (raw instruction probes) and chat-aware (prompts rendered through the model’s chat template)—that both produce baseline vectors, steered vectors, and deltas.

## Architecture at a Glance
- Entry points: `delta_activation_engine/cli.py` (baseline) and `delta_activation_engine/cli_chat.py` (chat-aware).
- Config loaders: `config/job.py` and `config/chat_job.py` parse strict YAML into dataclasses; no implicit defaults in dataclasses.
- Processing engine: `backends/hf.py::HFBackend` builds a HF model + RepE readers, selects middle-third layers, and computes last-layer/last-token averages.
- Pipelines: `pipelines/runner.py` (baseline) and `pipelines/chat_runner.py` (chat-aware) orchestrate prompt selection, backend calls, and artifact writes.
- Prompt sources: generic instruction probes (`prompts/probes_texts.py`) or wrapped chat prompts via `DeltaProbesPromptWrapper` + `DeltaProbesDataset`.
- Persistence: `io/files.py` writes compressed vectors and JSON metadata under a timestamped output directory.

## Initialization Flow (Ignition Key)
1) CLI parses `--config` → YAML → `DeltaActivationJobConfig` or `DeltaActivationChatJobConfig`.
2) For chat jobs, prompt configuration is also parsed into `PromptingConfig` (benchmark name, task type, probe list/source, optional thinking mode flag).
3) Baseline pipeline constructs `HFBackend(cfg)` explicitly; chat pipeline builds the same backend lazily unless injected (used in tests).
4) HF backend loads model/tokenizer via `setup_model_and_tokenizer`, sets `output_hidden_states=True`, detects control layers, and loads RepE readers for each emotion.
5) Pipelines derive the probe set (generic or provided), then dispatch baseline and steered forward passes.

## Processing Engine (What Runs)
- Representation: last hidden layer, last token, averaged across all prompts in the batch; max length fixed at 256.
- Control layers: middle third of decoder blocks (computed per model depth).
- Steering: `WrappedReadingVecModel` wraps decoder blocks and injects per-layer activation vectors scaled by intensity.
- Backend metadata includes selected layers and max length; emitted in `metadata.json` for reproducibility.

## Outputs and Storage
- `baseline.npz`: baseline vector.
- `deltas/emotion=<emo>_int=<intensity>.npz`: one delta per (emotion, intensity).
- `metadata.json`: provenance (model path, emotions, intensities, probe hash, timestamp, backend metadata, prompt_config for chat runs, chat template).
- Directory layout: `<output_dir>/<model>_<timestamp>/` for baseline; `<output_dir>/chat/<model>_<timestamp>/` for chat-aware.

## External Dependencies
- `neuro_manipulation` for HF model setup, layer detection, RepE reader loading, and wrapped reading-vec controller.
- `emotion_experiment_engine` for dataset/prompt abstractions (`BenchmarkConfig`, `BaseBenchmarkDataset`) used in chat pipeline prompt construction.

## Reliability Notes
- Probes in the baseline pipeline are raw templates with `{task}`/`{input}` placeholders; they are not formatted, mirroring legacy behavior.
- Errors surface early via strict YAML validation; missing keys raise `ValueError`, and basic type checks raise `TypeError`.
- Backends assume GPU-capable HF models; steering fails fast if no activation directions exist for a given emotion.
