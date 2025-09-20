"""
TrustLLM Robustness unit tests.

Responsible for: emotion_memory_experiments/datasets/trustllm_robustness.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(api_key="", OpenAI=object, AzureOpenAI=object)

sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))

from emotion_memory_experiments.data_models import BenchmarkConfig, ResultRecord
from emotion_memory_experiments.datasets.trustllm_robustness import TrustLLMRobustnessDataset


def _cfg(task: str, data_path: Path | None) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="trustllm_robustness",
        task_type=task,
        data_path=data_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
    )


def _write_json(tmp_path: Path, name: str, payload: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------
# Loader tests
# --------------------


def test_loader_advglue(tmp_path: Path):
    rows = [
        {"prompt": "Q", "index": 1, "task": "qqp", "label": 1, "type": "original"},
        {"prompt": "Q2", "index": 1, "task": "qqp", "label": 1, "type": "modified"},
    ]
    path = _write_json(tmp_path, "AdvGLUE.json", rows)
    ds = TrustLLMRobustnessDataset(config=_cfg("advglue", path), prompt_wrapper=None)
    assert len(ds) == 2
    assert ds[0]["ground_truth"]["label"] == 1


def test_loader_ood_detection(tmp_path: Path):
    rows = [{"prompt": "Do unsafe things"}]
    path = _write_json(tmp_path, "ood_detection.json", rows)
    ds = TrustLLMRobustnessDataset(config=_cfg("ood_detection", path), prompt_wrapper=None)
    assert len(ds) == 1
    assert ds[0]["ground_truth"] == "refuse"


def test_loader_rejects_bad_task(tmp_path: Path):
    path = _write_json(tmp_path, "AdvInstruction.json", [])
    try:
        TrustLLMRobustnessDataset(config=_cfg("adv_instruction", path), prompt_wrapper=None)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# --------------------
# Evaluator routing
# --------------------


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_eval_advinstruction_follow(mock_llm):
    mock_llm.return_value = {"label": "follow"}
    ds = TrustLLMRobustnessDataset(config=_cfg("advinstruction", None), prompt_wrapper=None)
    score = ds.evaluate_response("resp", ground_truth="follow", task_name="advinstruction", prompt="p")
    assert score == 1.0


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_eval_advglue_positive(mock_llm):
    mock_llm.return_value = {"label": "positive"}
    ds = TrustLLMRobustnessDataset(config=_cfg("advglue", None), prompt_wrapper=None)
    score = ds.evaluate_response(
        "resp",
        ground_truth={"task": "qqp", "label": 1},
        task_name="advglue",
        prompt="p",
    )
    assert score == 1.0


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_eval_ood_detection_refuse(mock_llm):
    mock_llm.return_value = {"label": "refuse"}
    ds = TrustLLMRobustnessDataset(config=_cfg("ood_detection", None), prompt_wrapper=None)
    score = ds.evaluate_response("resp", ground_truth="refuse", task_name="ood_detection", prompt="p")
    assert score == 1.0


# --------------------
# Split metrics & parity
# --------------------


def _record(task: str, gt: Any, resp: str, score: float, metadata: dict | None = None) -> ResultRecord:
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id=f"{task}_{gt}",
        task_name=task,
        prompt="p",
        response=resp,
        ground_truth=gt,
        score=score,
        repeat_id=0,
        metadata=metadata,
    )


def test_advglue_split_metrics_match_trustllm():
    from trustllm.utils import metrics as trust_metrics

    ds = TrustLLMRobustnessDataset(config=_cfg("advglue", None), prompt_wrapper=None)
    recs = [
        _record("advglue", {"task": "qqp", "label": 1}, "Response yes", 1.0, metadata={"index": 1, "task": "qqp", "type": "original"}),
        _record("advglue", {"task": "qqp", "label": 1}, "Response no", 0.0, metadata={"index": 1, "task": "qqp", "type": "modified"}),
        _record("advglue", {"task": "sst2", "label": 0}, "Negative", 1.0, metadata={"index": 2, "task": "sst2", "type": "original"}),
        _record("advglue", {"task": "sst2", "label": 0}, "Negative", 1.0, metadata={"index": 2, "task": "sst2", "type": "modified"}),
    ]
    metrics = ds.compute_split_metrics(recs)["advglue"]

    trust_input = []
    for r in recs:
        trust_input.append(
            {
                "task": (r.metadata or {}).get("task"),
                "index": (r.metadata or {}).get("index"),
                "type": (r.metadata or {}).get("type"),
                "eval_res": bool(r.score),
            }
        )
    trust_vals = trust_metrics.count_advglue(trust_input)
    assert abs(metrics["avg_acc"] - trust_vals["avg_acc"]) < 1e-9
    assert abs(metrics["RS"] - trust_vals["RS"]) < 1e-9


def test_ood_detection_split_metrics():
    from trustllm.utils import metrics as trust_metrics

    ds = TrustLLMRobustnessDataset(config=_cfg("ood_detection", None), prompt_wrapper=None)
    recs = [
        _record("ood_detection", "refuse", "I refuse", 1.0),
        _record("ood_detection", "refuse", "Sure", 0.0),
    ]
    metrics = ds.compute_split_metrics(recs)["ood_detection"]
    trust_val = trust_metrics.RtA(
        [{"eval_res": "LABEL_0" if r.score else "LABEL_1"} for r in recs]
    )
    assert abs(metrics["refusal_rate"] - trust_val) < 1e-9


def test_ood_generalization_split_metrics_simple():
    ds = TrustLLMRobustnessDataset(config=_cfg("ood_generalization", None), prompt_wrapper=None)
    recs = [
        _record("ood_generalization", "pancreatic neoplasm", "The diagnosis is pancreatic neoplasm.", 1.0, metadata={"source": "ddxplus"}),
        _record("ood_generalization", "stable angina", "Answer: stable angina", 0.0, metadata={"source": "ddxplus"}),
        _record("ood_generalization", "correct", "correct", 1.0, metadata={"source": "flipkart"}),
    ]
    metrics = ds.compute_split_metrics(recs)["ood_generalization"]
    assert metrics["per_source"]["ddxplus"] <= 1.0
    assert metrics["overall"] <= 1.0
