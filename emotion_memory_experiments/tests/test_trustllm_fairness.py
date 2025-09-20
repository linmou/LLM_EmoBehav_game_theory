"""
Single consolidated unit test file for TrustLLM Fairness tasks.
Covers: loader, evaluator routing (GPT-4o-mini JSON), split-level metrics,
and repo-import parity vs TrustLLM (excluding model forwarding).

Responsible for: emotion_memory_experiments/datasets/trustllm_fairness.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Optional
from unittest.mock import patch

# Ensure evaluation_utils can import without real OpenAI dependency
if "openai" not in sys.modules:
    # Provide both OpenAI and AzureOpenAI attributes to satisfy TrustLLM import
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=object, AzureOpenAI=object)

from emotion_memory_experiments.data_models import (
    BenchmarkConfig,
    ResultRecord,
)
from emotion_memory_experiments.datasets.trustllm_fairness import (
    TrustLLMFairnessDataset,
)


# --------------------
# Loader smoke tests
# --------------------

def _make_cfg(task_type: str, data_path: Optional[str]):
    return BenchmarkConfig(
        name="trustllm_fairness",
        task_type=task_type,
        data_path=data_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
    )


def test_loader_stereotype_recognition_minimal_fixture(tmp_path):
    data = [
        {"prompt": "Text1", "label": "stereotype"},
        {"prompt": "Text2", "label": "anti-stereotype"},
        {"prompt": "Text3", "label": "unrelated"},
    ]
    p = tmp_path / "fairness_recognition_sample.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    ds = TrustLLMFairnessDataset(
        config=_make_cfg("stereotype_recognition", str(p)), prompt_wrapper=None
    )
    assert len(ds) == 3
    it0 = ds[0]
    assert it0["prompt"].startswith("Text1")
    assert it0["ground_truth"] == "stereotype"


def test_loader_stereotype_agreement_minimal_fixture(tmp_path):
    data = [
        {"prompt": "Agree?", "label": "yes"},
        {"prompt": "Agree?", "label": "no"},
    ]
    p = tmp_path / "fairness_agreement_sample.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    ds = TrustLLMFairnessDataset(
        config=_make_cfg("stereotype_agreement", str(p)), prompt_wrapper=None
    )
    assert len(ds) == 2
    assert ds[1]["ground_truth"].lower() in ("yes", "no")


# --------------------
# Evaluator routing (mocked OpenAI)
# --------------------


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_eval_stereotype_recognition_label_match(mock_llm):
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"label": "anti-stereotype"}
    cfg = _make_cfg("stereotype_recognition", None)
    ds = TrustLLMFairnessDataset(config=cfg, prompt_wrapper=None)
    score = ds.evaluate_response(
        "resp",
        ground_truth="anti-stereotype",
        task_name="stereotype_recognition",
        prompt="p",
    )
    assert score == 1.0


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_eval_stereotype_agreement_yes_no(mock_llm):
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"label": "yes"}
    cfg = _make_cfg("stereotype_agreement", None)
    ds = TrustLLMFairnessDataset(config=cfg, prompt_wrapper=None)
    score = ds.evaluate_response(
        "resp", ground_truth="yes", task_name="stereotype_agreement", prompt="p"
    )
    assert score == 1.0


# --------------------
# Split-level metrics
# --------------------


def _rec(task: str, gt: str, correct: bool) -> ResultRecord:
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id=f"{gt}",
        task_name=task,
        prompt="p",
        response="r",
        ground_truth=gt,
        score=1.0 if correct else 0.0,
        repeat_id=0,
    )


def test_split_metrics_stereotype_recognition_macro_counts():
    cfg = _make_cfg("stereotype_recognition", None)
    ds = TrustLLMFairnessDataset(config=cfg, prompt_wrapper=None)
    # stereotype: 2/3, anti: 1/2, unrelated: 0/1
    recs = [
        _rec("stereotype_recognition", "stereotype", True),
        _rec("stereotype_recognition", "stereotype", True),
        _rec("stereotype_recognition", "stereotype", False),
        _rec("stereotype_recognition", "anti-stereotype", True),
        _rec("stereotype_recognition", "anti-stereotype", False),
        _rec("stereotype_recognition", "unrelated", False),
    ]
    m = ds.compute_split_metrics(recs)
    fm = m["stereotype_recognition"]
    assert fm["counts"]["stereotype"] == 3
    assert fm["counts"]["anti-stereotype"] == 2
    assert fm["counts"]["unrelated"] == 1
    assert abs(fm["per_class_acc"]["stereotype"] - (2 / 3)) < 1e-9
    assert abs(fm["per_class_acc"]["anti-stereotype"] - 0.5) < 1e-9
    assert abs(fm["per_class_acc"]["unrelated"] - 0.0) < 1e-9
    expected_macro = ((2 / 3) + 0.5 + 0.0) / 3
    assert abs(fm["macro_acc"] - expected_macro) < 1e-9


def test_split_metrics_stereotype_agreement_micro():
    cfg = _make_cfg("stereotype_agreement", None)
    ds = TrustLLMFairnessDataset(config=cfg, prompt_wrapper=None)
    recs = [
        _rec("stereotype_agreement", "yes", True),
        _rec("stereotype_agreement", "no", False),
        _rec("stereotype_agreement", "yes", True),
    ]
    m = ds.compute_split_metrics(recs)
    assert "overall" in m
    assert abs(m["overall"] - (2 / 3)) < 1e-9


# --------------------
# Repo-import parity vs TrustLLM (excluding model forwarding)
# --------------------


def _cfg(task: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="trustllm_fairness",
        task_type=task,
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )


def test_repo_parity_stereotype_agreement_micro():
    # Make TrustLLM available and stub OpenAI
    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=object, AzureOpenAI=object)
    from trustllm.task.fairness import FairnessEval  # type: ignore

    items = [
        {"res": "yes", "label": "yes"},
        {"res": "no", "label": "yes"},
        {"res": "yes", "label": "yes"},
    ]
    trust = FairnessEval()
    trust_overall = trust.stereotype_agreement_eval(items)

    recs = []
    for i, it in enumerate(items):
        correct = it["res"].strip().lower() == it["label"].strip().lower()
        recs.append(
            ResultRecord(
                emotion="anger",
                intensity=1.0,
                item_id=str(i),
                task_name="stereotype_agreement",
                prompt="p",
                response=it["res"],
                ground_truth=it["label"],
                score=1.0 if correct else 0.0,
                repeat_id=0,
            )
        )

    ds = TrustLLMFairnessDataset(config=_cfg("stereotype_agreement"), prompt_wrapper=None)
    our_overall = ds.compute_split_metrics(recs)["overall"]
    assert abs(our_overall - trust_overall) < 1e-9


def test_repo_parity_stereotype_recognition_macro():
    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=object, AzureOpenAI=object)
    from trustllm.task.fairness import FairnessEval  # type: ignore

    items = [
        {"res": "it's a stereotype", "label": "stereotype"},
        {"res": "it's an anti-stereotype", "label": "anti-stereotype"},
        {"res": "it's unrelated and meaningless", "label": "unrelated"},
        {"res": "it's a stereotype", "label": "unrelated"},
    ]
    trust = FairnessEval()
    trust_overall = trust.stereotype_recognition_eval(items)

    mapping = {
        "it's a stereotype": "stereotype",
        "it's an anti-stereotype": "anti-stereotype",
        "it's unrelated and meaningless": "unrelated",
    }
    recs = []
    for i, it in enumerate(items):
        pred = mapping.get(it["res"], "")
        correct = pred == it["label"]
        recs.append(
            ResultRecord(
                emotion="anger",
                intensity=1.0,
                item_id=str(i),
                task_name="stereotype_recognition",
                prompt="p",
                response=it["res"],
                ground_truth=it["label"],
                score=1.0 if correct else 0.0,
                repeat_id=0,
            )
        )

    ds = TrustLLMFairnessDataset(config=_cfg("stereotype_recognition"), prompt_wrapper=None)
    m = ds.compute_split_metrics(recs)
    our_overall = m["overall"]
    assert abs(our_overall - trust_overall) < 1e-9
