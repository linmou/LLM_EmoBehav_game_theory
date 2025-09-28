# Tests for emotion_experiment_engine/experiment.py neutral-only workflow

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_experiment_engine.data_models import (
    BenchmarkConfig,
    ExperimentConfig,
    ResultRecord,
    VLLMLoadingConfig,
)
from emotion_experiment_engine.experiment import EmotionExperiment


class _DummyDataset:
    def __init__(self) -> None:
        self.items = [
            {
                "prompts": ["dummy prompt"],
                "items": [MagicMock(id="item-1", metadata={})],
                "ground_truths": ["gt"],
            }
        ]
        self.collate_fn = lambda batch: batch

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 1


@pytest.fixture
def experiment_config(tmp_path: Path) -> ExperimentConfig:
    data_path = tmp_path / "dummy.jsonl"
    data_path.write_text("{}\n")

    benchmark = BenchmarkConfig(
        name="dummy_benchmark",
        task_type="dummy_task",
        data_path=data_path,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )

    loading_config = VLLMLoadingConfig(
        model_path="dummy-model",
        gpu_memory_utilization=0.5,
        tensor_parallel_size=None,
        max_model_len=1024,
        enforce_eager=True,
        quantization=None,
        trust_remote_code=True,
        dtype="float16",
        seed=123,
        disable_custom_all_reduce=False,
        additional_vllm_kwargs={},
    )

    return ExperimentConfig(
        model_path="dummy-model",
        emotions=[],
        intensities=[0.0],
        benchmark=benchmark,
        output_dir=str(tmp_path),
        batch_size=1,
        generation_config={"enable_thinking": False},
        loading_config=loading_config,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
    )


@pytest.fixture(autouse=True)
def patch_heavy_dependencies(monkeypatch: pytest.MonkeyPatch):
    dummy_dataset = _DummyDataset()

    def fake_create_components(*args: Any, **kwargs: Any):
        return (MagicMock(), MagicMock(), dummy_dataset)

    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.create_benchmark_components",
        fake_create_components,
    )

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.vocab_size = 10
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 2

    class _DummyPromptFormat:
        def __init__(self, tokenizer: Any):
            self.tokenizer = tokenizer

    monkeypatch.setattr(
        "neuro_manipulation.prompt_formats.PromptFormat",
        _DummyPromptFormat,
    )

    def fake_load_tokenizer_only(*args: Any, **kwargs: Any):
        return _FakeTokenizer(), None

    monkeypatch.setattr(
        "neuro_manipulation.utils.load_tokenizer_only",
        fake_load_tokenizer_only,
    )

    def fake_setup_model_and_tokenizer(*args: Any, **kwargs: Any):
        model = MagicMock()
        model.device = "cpu"
        tokenizer = _FakeTokenizer()
        prompt_format = MagicMock()
        processor = None
        return model, tokenizer, prompt_format, processor

    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.setup_model_and_tokenizer",
        fake_setup_model_and_tokenizer,
    )

    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.get_repe_eng_config",
        lambda *a, **k: {"block_name": "dummy", "control_method": "add"},
    )

    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.ModelLayerDetector.num_layers",
        classmethod(lambda cls, model: 2),
    )

    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.get_pipeline",
        lambda *a, **k: MagicMock(return_value=[{"generated_text": "out"}]),
    )

    yield


def test_neutral_config_skips_emotion_reader_loading(
    monkeypatch: pytest.MonkeyPatch, experiment_config: ExperimentConfig
) -> None:
    def _raise_if_called(*args: Any, **kwargs: Any):
        raise AssertionError("load_emotion_readers should not be invoked when emotions list is empty")

    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.load_emotion_readers",
        _raise_if_called,
    )

    # No exception should be raised during instantiation if readers are skipped.
    EmotionExperiment(experiment_config)


def test_neutral_only_run_emits_results_without_activations(
    monkeypatch: pytest.MonkeyPatch, experiment_config: ExperimentConfig, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.load_emotion_readers",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected reader load")),
    )

    activations_seen: List[Any] = []

    def fake_forward(self: EmotionExperiment, data_loader: Any, activations: Dict[str, Any] | None):
        activations_seen.append(activations)
        return [
            ResultRecord(
                emotion=self.cur_emotion,
                intensity=self.cur_intensity,
                item_id="item-1",
                task_name="dummy",
                prompt="p",
                response="r",
                ground_truth="gt",
                score=1.0,
                repeat_id=0,
            )
        ]

    monkeypatch.setattr(
        EmotionExperiment,
        "_forward_dataloader",
        fake_forward,
    )

    monkeypatch.setattr(
        EmotionExperiment,
        "_save_results",
        lambda self, results: pd.DataFrame([r.__dict__ for r in results]),
    )

    experiment = EmotionExperiment(experiment_config)
    # ensure the neutral dataset resolves to deterministic loader output
    monkeypatch.setattr(
        experiment,
        "build_dataloader",
        lambda emotion: [
            {
                "prompts": ["dummy"],
                "items": [MagicMock(id="item-1", metadata={})],
                "ground_truths": ["gt"],
            }
        ],
    )

    df = experiment.run_experiment()

    assert activations_seen == [None]
    assert df.shape[0] == 1
    assert df.iloc[0]["emotion"] == "neutral"


def test_series_runner_accepts_empty_emotions(monkeypatch: pytest.MonkeyPatch, experiment_config: ExperimentConfig) -> None:
    created_configs: List[ExperimentConfig] = []

    class _RecordingExperiment:
        def __init__(self, config: ExperimentConfig, *args: Any, **kwargs: Any):
            created_configs.append(config)
            self.config = config
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger = MagicMock()
            self.emotion_datasets = {}

        def run_experiment(self):
            return MagicMock()

    monkeypatch.setattr(
        "emotion_experiment_engine.experiment.EmotionExperiment",
        _RecordingExperiment,
    )

    from emotion_experiment_engine.emotion_experiment_series_runner import (
        MemoryExperimentSeriesRunner,
    )

    config_path = experiment_config.benchmark.data_path
    assert config_path is not None

    base_yaml = {
        "models": [experiment_config.model_path],
        "benchmarks": [
            {
                "name": experiment_config.benchmark.name,
                "task_type": experiment_config.benchmark.task_type,
                "data_path": str(config_path),
            }
        ],
        "emotions": [],
        "intensities": experiment_config.intensities,
        "loading_config": {
            "model_path": experiment_config.model_path,
            "gpu_memory_utilization": experiment_config.loading_config.gpu_memory_utilization,
            "tensor_parallel_size": experiment_config.loading_config.tensor_parallel_size,
            "max_model_len": experiment_config.loading_config.max_model_len,
            "enforce_eager": experiment_config.loading_config.enforce_eager,
            "quantization": experiment_config.loading_config.quantization,
            "trust_remote_code": experiment_config.loading_config.trust_remote_code,
            "dtype": experiment_config.loading_config.dtype,
            "seed": experiment_config.loading_config.seed,
            "disable_custom_all_reduce": experiment_config.loading_config.disable_custom_all_reduce,
            "additional_vllm_kwargs": experiment_config.loading_config.additional_vllm_kwargs,
        },
        "execution": {
            "repeat_runs": 1,
        },
        "output_dir": str(config_path.parent / "neutral_outputs"),
    }

    yaml_path = config_path.parent / "neutral_config.yaml"
    yaml_path.write_text(json.dumps(base_yaml))

    runner = MemoryExperimentSeriesRunner(str(yaml_path))

    benchmark_cfg = runner.base_config["benchmarks"][0]
    model_name = runner.base_config["models"][0]
    runner.setup_experiment(benchmark_cfg, model_name)

    assert created_configs[0].emotions == []
