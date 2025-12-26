# HFBackend (`delta_activation_engine/backends/hf.py`)
Last updated: 2025-12-01 (working copy)

## Purpose
HF-backed processor that produces baseline and steered representations using RepE activation directions. This is the computational core for both baseline and chat pipelines.

## Implementation Walkthrough
- `select_middle_third_layers(total_layers)`: returns the middle third of decoder layers; empty list if depth ≤ 0.
- `HFBackend.__init__(cfg)`: lazily imports HF/RepE utilities, loads model/tokenizer via `setup_model_and_tokenizer(from_vllm=False)`, forces `output_hidden_states=True`, detects layer count, and selects control layers. It registers RepE pipelines, builds RepE config (`get_repe_eng_config`), loads emotion readers, and wraps the model with `WrappedReadingVecModel`. Max sequence length is fixed at 256.
- RepE reader loading honors `emotion_data_seed` in `repe_eng_config` when hashing caches and building train/test splits. Adjust this seed to reshuffle RepE directions between delta runs.
- `_forward_last_hidden_avg(texts)`: tokenizes with padding/truncation, runs HF forward pass, takes last hidden state → last token slice → mean over batch → CPU numpy float32.
- `_forward_last_hidden_avg_steered(texts, emotion, intensity)`: fetches emotion directions for control layers, scales by intensity, wraps decoder blocks, sets controllers (no normalization, mask=1.0), runs forward, averages last token, then resets wrapper.
- `get_repr(prompts, steered, emotion=None, intensity=None)`: routes to baseline vs steered paths, asserting emotion/intensity for steered runs.
- `get_run_metadata()`: returns backend name, control layers, and max length.

## Key Logic
- Layer selection keeps steering in the middle of the network to avoid early noise and late bottleneck effects.
- Representations are last-layer/last-token to align with prior delta activation baselines and tests.
- Steering is additive: delta is derived by subtracting baseline vectors outside the backend.

## Dependencies
- `neuro_manipulation.model_utils.setup_model_and_tokenizer`
- `neuro_manipulation.repe.rep_control_reading_vec.WrappedReadingVecModel`
- `neuro_manipulation.model_layer_detector.ModelLayerDetector`
- RepE config/readers via `neuro_manipulation.repe` modules.

## Potential Issues / Gaps
- Assumes RepE directions exist for each requested emotion; raises `ValueError` otherwise.
- Max length is hard-coded (256); long prompts will truncate silently.
- Uses last-token pooling only; no option for mean pooling or token_pos targeting.
- Control layers computed once; not configurable per run.

## Usage Example
```python
from delta_activation_engine.backends.hf import HFBackend
from delta_activation_engine.config import load_job_config_from_yaml

cfg = load_job_config_from_yaml("config/delta_activations/qwen2_5b_default.yaml")
backend = HFBackend(cfg)
vec = backend.get_repr(["Say hello"], steered=False)
vec_steered = backend.get_repr(["Say hello"], steered=True, emotion="anger", intensity=1.0)
```
