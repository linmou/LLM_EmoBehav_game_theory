"""
Consolidated tests for repeat support + CLI/series integration + README output.

Responsible files:
- emotion_experiment_engine/experiment.py
- emotion_experiment_engine/run_emotion_memory_experiment.py
- emotion_experiment_engine/memory_experiment_series_runner.py
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from emotion_experiment_engine.data_models import (
    BenchmarkConfig,
    ExperimentConfig,
    ResultRecord,
)
from emotion_experiment_engine.experiment import EmotionExperiment
from emotion_experiment_engine.run_emotion_memory_experiment import run_experiment


def _make_dummy_tokenizer():
    class DummyTok:
        vocab_size = 10
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2
        unk_token_id = 3

    return DummyTok()


class CapturePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts, activations=None, **kwargs):
        self.calls.append(kwargs)
        return [{"generated_text": p + " out"} for p in prompts]


def test_repeat_aggregation_and_readme():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        bench = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=out_dir / "dummy.jsonl",
            base_data_dir=str(out_dir),
            sample_limit=3,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )
        (out_dir / "dummy.jsonl").write_text("")

        cfg = ExperimentConfig(
            model_path="/fake/model",
            emotions=["anger"],
            intensities=[1.0],
            benchmark=bench,
            output_dir=str(out_dir),
            batch_size=1,
            generation_config=None,
            loading_config=None,
            repe_eng_config=None,
            max_evaluation_workers=1,
            pipeline_queue_size=1,
        )

        with patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(_make_dummy_tokenizer(), None)), \
             patch("emotion_experiment_engine.experiment.create_benchmark_components", return_value=(None, None, [])), \
             patch.object(EmotionExperiment, "_build_emotion_datasets", return_value={}), \
             patch("neuro_manipulation.prompt_formats.PromptFormat", lambda tok: MagicMock()):
            exp = EmotionExperiment(cfg, dry_run=True)

        # Prepare two repeats
        results = []
        for idx, score in enumerate([1.0, 1.0, 0.0]):
            results.append(
                ResultRecord(
                    emotion="anger",
                    intensity=1.0,
                    item_id=f"a0_{idx}",
                    task_name="passkey",
                    prompt=f"p{idx}",
                    response=f"r{idx}",
                    ground_truth="gt",
                    score=score,
                    repeat_id=0,
                    metadata={"benchmark": "infinitebench"},
                )
            )
        for idx, score in enumerate([0.0, 1.0, 0.0]):
            results.append(
                ResultRecord(
                    emotion="anger",
                    intensity=1.0,
                    item_id=f"a1_{idx}",
                    task_name="passkey",
                    prompt=f"p{idx}",
                    response=f"r{idx}",
                    ground_truth="gt",
                    score=score,
                    repeat_id=1,
                    metadata={"benchmark": "infinitebench"},
                )
            )

        df = exp._save_results(results)
        files = {p.name: p for p in Path(exp.output_dir).iterdir()}
        assert "detailed_results.csv" in files
        assert "summary_results.csv" in files
        assert "summary_by_repeat.csv" in files
        assert "summary_overall.csv" in files
        assert (Path(exp.output_dir) / "README.md").exists()


def test_cli_passes_repeat_runs_and_seed_to_experiment(tmp_path: Path):
    data_file = tmp_path / "dummy.jsonl"
    data_file.write_text("{}\n")

    cfg = {
        "model": {"model_path": "/mock/model"},
        "emotions": {"target_emotions": ["anger"], "intensities": [1.0]},
        "benchmarks": [
            {
                "name": "infinitebench",
                "task_type": "passkey",
                "data_path": str(data_file),
            }
        ],
        "generation": {"do_sample": True, "temperature": 0.7},
        "execution": {"batch_size": 2, "repeat_runs": 3, "repeat_seed_base": 1234},
        "output": {"results_dir": str(tmp_path)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    class DummyExperiment:
        def __init__(self, exp_config, dry_run=False, repeat_runs=1, repeat_seed_base=None):
            self.exp_config = exp_config
            self.dry_run = dry_run
            self.repeat_runs = repeat_runs
            self.repeat_seed_base = repeat_seed_base

        def run_experiment(self):
            return pd.DataFrame({
                "emotion": ["anger"],
                "benchmark": [self.exp_config.benchmark.name],
                "score": [0.5],
            })

    with patch("emotion_experiment_engine.experiment.EmotionExperiment", DummyExperiment):
        ok = run_experiment(cfg_path, dry_run=False, debug=False)
        assert ok is True


def test_per_run_seed_passed_in_generation_kwargs(tmp_path: Path):
    bench = BenchmarkConfig(
        name="infinitebench",
        task_type="passkey",
        data_path=tmp_path / "dummy.jsonl",
        base_data_dir=str(tmp_path),
        sample_limit=2,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )
    (tmp_path / "dummy.jsonl").write_text("")

    cfg = ExperimentConfig(
        model_path="/mock/model",
        emotions=["anger"],
        intensities=[1.0],
        benchmark=bench,
        output_dir=str(tmp_path),
        batch_size=2,
        generation_config={"do_sample": True, "temperature": 0.7, "max_new_tokens": 8},
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
    )

    with patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(_make_dummy_tokenizer(), None)), \
         patch("emotion_experiment_engine.experiment.create_benchmark_components", return_value=(None, None, [])), \
         patch("neuro_manipulation.prompt_formats.PromptFormat", lambda tok: MagicMock()):
        exp = EmotionExperiment(cfg, dry_run=True, repeat_runs=1)

    exp.is_vllm = False
    exp.rep_control_pipeline = CapturePipeline()
    exp.dataset = MagicMock()
    exp.dataset.evaluate_batch.return_value = [0.0, 0.0]
    exp.config.benchmark.task_type = "passkey"
    exp.batch_size = 2

    batch = {
        "prompts": ["p1", "p2"],
        "items": [MagicMock(id=1), MagicMock(id=2)],
        "ground_truths": ["gt1", "gt2"],
    }
    data_loader = [batch]

    exp.repeat_seed_base = 1000
    exp.cur_repeat = 2

    _ = exp._forward_dataloader(data_loader, activations={})

    assert len(exp.rep_control_pipeline.calls) == 1
    kwargs = exp.rep_control_pipeline.calls[0]
    assert kwargs.get("random_seed") == 1002


def test_series_runner_passes_repeat_and_seed(tmp_path: Path):
    cfg = {
        "models": ["/mock/model"],
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmarks": [
            {
                "name": "infinitebench",
                "task_type": "passkey",
                "sample_limit": 1,
                "enable_auto_truncation": False,
                "truncation_strategy": "right",
                "preserve_ratio": 0.8,
            }
        ],
        "output_dir": str(tmp_path / "out"),
        "loading_config": {
            "model_path": "/mock/model",
            "gpu_memory_utilization": 0.8,
            "tensor_parallel_size": 1,
            "max_model_len": 1024,
            "enforce_eager": True,
            "quantization": None,
            "trust_remote_code": True,
            "dtype": "float16",
            "seed": 42,
            "disable_custom_all_reduce": False,
            "additional_vllm_kwargs": {},
        },
        "repeat_runs": 4,
        "repeat_seed_base": 777,
    }

    cfg_path = tmp_path / "series.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    captured = {}

    class DummyExp:
        def __init__(self, exp_config, dry_run=False, repeat_runs=1, repeat_seed_base=None):
            captured["repeat_runs"] = repeat_runs
            captured["repeat_seed_base"] = repeat_seed_base
            self.output_dir = tmp_path / "out" / "dummy"
            self.emotion_datasets = {}

        def run_experiment(self):
            return None

    with patch("emotion_experiment_engine.experiment.EmotionExperiment", DummyExp):
        from emotion_experiment_engine.memory_experiment_series_runner import (
            MemoryExperimentSeriesRunner,
        )

        runner = MemoryExperimentSeriesRunner(str(cfg_path), dry_run=True)
        bench = cfg["benchmarks"][0]
        _ = runner.setup_experiment(bench, cfg["models"][0])

    assert captured["repeat_runs"] == 4
    assert captured["repeat_seed_base"] == 777

