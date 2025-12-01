# Components: Config and Tests

This document covers the remaining critical pieces that are not core Python modules: the benchmark configuration and the top-level smoke tests.

## Benchmark Config: `config/pd_behavior_game_theory.yaml`

### Purpose

Defines how the `emotion_experiment_engine` should construct the `game_theory` / `Prisoners_Dilemma` benchmark for PD behavior transfer experiments.

### Structure

```yaml
name: "game_theory"
task_type: "Prisoners_Dilemma"

sample_limit: null
augmentation_config: null

enable_auto_truncation: false
truncation_strategy: "right"
preserve_ratio: 1.0
llm_eval_config: null
base_data_dir: null
data_path: null

generation_config:
  max_new_tokens: 256
  temperature: 0.0
  top_p: 1.0
  do_sample: false
  repetition_penalty: 1.0

batch_size: 100
```

### Usage

- Parsed by `_load_benchmark_config` and `_load_generation_config` in `run_pd_defection_pd_behavior.py`.  
- Controls:
  - Which benchmark to run (`name`, `task_type`).  
  - Truncation behavior.  
  - Generation hyperparameters used by `_compute_defect_ratio`.

If you add new benchmark settings, keep `_load_benchmark_config` and the tests in sync.

## Activation Specs: `config/pd_defection_*.json`

### Purpose

Capture metadata for a particular PD defection vector to be reused in behavior experiments.

Example (`pd_defection_iter8_qwen2.5_0.5B.json`):

```json
{
  "pd_result_dir": "auto_experiments/task_similarity/results/Qwen2.5-0.5B-Instruct_20251129_211403",
  "layer": 8,
  "vector_path": "best_vector.npy",
  "span_mode": "option",
  "pd_best_layer": 8,
  "pd_best_accuracy": 0.5515055467511886,
  "pd_seed": 0,
  "pd_max_pairs": null
}
```

### Usage

- Parsed by `_load_activation_spec` into a `PDActivationSpec`.  
- Relies on:
  - `pd_result_dir` being a valid `run_pd_defection_experiment` output directory.  
  - `vector_path` pointing to a defection vector file (relative to `pd_result_dir` if not absolute).

When creating new specs, keep the structure consistent so that tooling and scripts remain simple.

## Smoke Test: `tests/test_pd_run_smoke.py`

### Purpose

High-level smoke test for `run_pd_defection_experiment.run` that:

- Avoids loading real HF models or PD data.  
- Verifies that:
  - Result structure is consistent.  
  - Vectors and metrics are written to disk.  
  - The best layer selection logic works.

### Approach

- Patches:
  - `build_pd_pair_bundle` to return a synthetic `PDPairBundle`.  
  - `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained` to return dummy classes.  
  - `train_pd_repreader` to return synthetic accuracies and vectors.  
  - `_decision_rate`, `_register_control_hook`, `_token_id` to simple stubs.

- Invokes `mod.run(...)` with small arguments and asserts:
  - `best_layer` and `best_accuracy` are present and consistent.  
  - A run directory `dummy-model_*` was created with `result.json` and `best_vector.npy`.  
  - Layer vectors are saved under `layer_vectors/dummy-model/`.

This test is a good template if you need to extend behavior while keeping dependencies mocked during development.

## Smoke Test: `tests/test_pd_behavior_run_smoke.py`

### Purpose

High-level smoke test for `run_pd_defection_pd_behavior.run` that:

- Validates wiring between activation specs, benchmark config, and steering logic.  
- Ensures that steering intensity has a predictable effect on the (dummy) defection ratio.

### Approach

- Defines:
  - `_DummyTokenizer` that records intensity via special token IDs.  
  - `_DummyModel` whose `generate` method emits token IDs encoding the current intensity.  
  - `_DummyDataset` that mimics `GameTheoryDataset` and reuses its static parsing helpers.

- Patches in the module:
  - `AutoTokenizer` / `AutoModelForCausalLM`.  
  - `PromptFormat`.  
  - `GameTheoryDataset`.  
  - `_register_control_hook` to update a global `_CURRENT_INTENSITY`.

- Builds a minimal benchmark config and activation spec on disk.  
- Calls `mod.run(...)` with intensities `[0.0, 1.0]`.  
- Asserts:
  - Result metadata fields are populated.  
  - `defect_ratio[0.0] == 0.0` and `defect_ratio[1.0] == 1.0` in the dummy pipeline.  
  - At least one behavior summary JSON file exists under the output directory.

This test gives a compact example of how to drive the behavior pipeline end-to-end without invoking the real benchmark engine or HF stack.

