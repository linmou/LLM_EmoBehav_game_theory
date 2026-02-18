"""
Tests for registry wiring and HumanEval dataset modes.

This suite intentionally avoids network and relies on local paths:
- HUMANEVAL_ORIG_GZ: path to HumanEval.jsonl(.gz)
- HUMANEVAL_PLUS_GZ: path to HumanEvalPlus.jsonl.gz (optional; tests skip if missing)
"""

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig


class DummyPF:
    def build(self, system_prompt, user_messages, assistant_messages=None, images=None, enable_thinking=False):
        return "\n".join([system_prompt] + (user_messages or []))


def make_cfg(name: str, task: str, path: Path, sample_limit: int = 3) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=name,
        task_type=task,
        data_path=path,
        base_data_dir=None,
        sample_limit=sample_limit,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_registry_unknown_task_raises():
    pf = DummyPF()
    cfg = make_cfg("humaneval", "unknown_task", Path("/tmp/does_not_exist"))
    with pytest.raises(KeyError):
        create_benchmark_components(
            benchmark_name="humaneval", task_type="unknown_task", config=cfg, prompt_format=pf
        )


def test_humaneval_default_loads_local_orig():
    # prefer env var; else fallback to a common local path
    path_str = os.environ.get(
        "HUMANEVAL_ORIG_GZ", os.path.expandvars("${USER_HOME}/human-eval/data/HumanEval.jsonl.gz").replace("${USER_HOME}", "/home/jjl7137")
    )
    path = Path(path_str)
    if not path.exists():
        pytest.skip("HumanEval original file not found; set HUMANEVAL_ORIG_GZ to run this test")

    pf = DummyPF()
    cfg = make_cfg("humaneval", "default", path)
    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name="humaneval", task_type="default", config=cfg, prompt_format=pf
    )
    assert len(dataset) > 0
    item = dataset[0]["item"]
    assert item.metadata["mode"] == "default"
    assert item.metadata["source"] == "humaneval"


def test_humaneval_star_emits_default_and_plus():
    plus_path = os.environ.get("HUMANEVAL_PLUS_GZ")
    if not plus_path or not Path(plus_path).exists():
        pytest.skip("HUMANEVAL_PLUS_GZ not set; skipping star mode test")

    pf = DummyPF()
    cfg = make_cfg("humaneval", "*", Path(plus_path))
    _, _, dataset = create_benchmark_components(
        benchmark_name="humaneval", task_type="*", config=cfg, prompt_format=pf
    )
    assert len(dataset) > 0
    # Expect at least one ::default item
    ids = [it["item"].id for it in [dataset[i] for i in range(min(len(dataset), 10))]]
    assert any("::default" in s for s in ids)


def test_humaneval_default_canonical_solution_passes():
    path_str = os.environ.get(
        "HUMANEVAL_ORIG_GZ", os.path.expandvars("${USER_HOME}/human-eval/data/HumanEval.jsonl.gz").replace("${USER_HOME}", "/home/jjl7137")
    )
    path = Path(path_str)
    if not path.exists():
        pytest.skip("HumanEval original file not available; set HUMANEVAL_ORIG_GZ to run")

    pf = DummyPF()
    cfg = make_cfg("humaneval", "default", path, sample_limit=1)
    _, _, dataset = create_benchmark_components(
        benchmark_name="humaneval", task_type="default", config=cfg, prompt_format=pf
    )
    sample = dataset[0]
    gt = sample["ground_truth"]
    canonical = gt["canonical_solution"]
    score = dataset.evaluate_response(canonical, gt, task_name="default", prompt=sample["prompt"])
    assert score == pytest.approx(1.0)

    bad_completion = "pass"
    score_bad = dataset.evaluate_response(bad_completion, gt, task_name="default", prompt=sample["prompt"])
    assert score_bad == pytest.approx(0.0)


def test_humaneval_plus_canonical_solution_passes():
    plus_path = os.environ.get("HUMANEVAL_PLUS_GZ")
    if not plus_path or not Path(plus_path).exists():
        pytest.skip("HUMANEVAL_PLUS_GZ not set; skipping plus evaluation test")

    pf = DummyPF()
    cfg = make_cfg("humaneval", "plus", Path(plus_path), sample_limit=1)
    _, _, dataset = create_benchmark_components(
        benchmark_name="humaneval", task_type="plus", config=cfg, prompt_format=pf
    )
    sample = dataset[0]
    gt = sample["ground_truth"]
    canonical = gt["canonical_solution"]
    score = dataset.evaluate_response(canonical, gt, task_name="plus", prompt=sample["prompt"])
    assert score == pytest.approx(1.0)

    bad_completion = "def {}():\n    return 0".format(gt["entry_point"]) if gt.get("entry_point") else "return 0"
    score_bad = dataset.evaluate_response(bad_completion, gt, task_name="plus", prompt=sample["prompt"])
    assert score_bad == pytest.approx(0.0)
