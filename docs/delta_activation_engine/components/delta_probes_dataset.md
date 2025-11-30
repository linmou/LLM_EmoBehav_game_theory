# DeltaProbesDataset (`delta_activation_engine/datasets/probes.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Adapts a list of probe strings into `BenchmarkItem` records compatible with `emotion_experiment_engine` pipelines, enabling prompt rendering via wrappers/PromptFormat.

## Implementation Walkthrough
- Constructor stores a copy of `probes` and forwards config/prompt wrapper/truncation settings to `BaseBenchmarkDataset`.
- `_load_and_parse_data`: iterates probes, building `BenchmarkItem` objects with incrementing integer IDs, input text set to the probe string, empty `ground_truth`, and `metadata=None`.
- `evaluate_response`: returns `0.0` because probes are not scored; placeholder to satisfy abstract contract.
- `get_task_metrics`: returns an empty list (no metrics defined).

## Key Logic
- Dataset length equals number of probes; `__getitem__` inherited from base class yields dicts that include rendered prompts via the provided wrapper.
- Stateless evaluation—responses are not graded; the dataset exists purely to drive prompt construction.

## Dependencies
- `emotion_experiment_engine` data models: `BenchmarkConfig`, `BenchmarkItem`.
- Inherits from `BaseBenchmarkDataset`.

## Potential Issues / Gaps
- No validation of probe text contents; placeholders stay intact.
- Evaluation and metrics are stubbed; callers should not expect scores.

## Usage Example
```python
from delta_activation_engine.datasets.probes import DeltaProbesDataset
from delta_activation_engine.prompts.wrappers import DeltaProbesPromptWrapper
from neuro_manipulation.prompt_formats import PromptFormat
from emotion_experiment_engine.data_models import BenchmarkConfig
from neuro_manipulation.utils import load_tokenizer_only

tokenizer, _ = load_tokenizer_only(model_name_or_path="/models/DUMMY", expand_vocab=False, auto_load_multimodal=True)
pf = PromptFormat(tokenizer)
wrapper = DeltaProbesPromptWrapper(pf)
config = BenchmarkConfig(name="delta_probes", task_type="default", data_path=None, base_data_dir=None, sample_limit=None, augmentation_config=None, enable_auto_truncation=False, truncation_strategy="right", preserve_ratio=1.0, llm_eval_config=None)
dataset = DeltaProbesDataset(config=config, prompt_wrapper=wrapper, probes=["Say hello"], tokenizer=tokenizer)
record = dataset[0]
print(record["prompt"])
```
