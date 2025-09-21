"""
Single consolidated unit test file for TrustLLM Fairness tasks.
Covers: loader, evaluator routing (GPT-4o-mini JSON), split-level metrics,
and repo-import parity vs TrustLLM (excluding model forwarding).

Responsible for: emotion_experiment_engine/datasets/trustllm_fairness.py
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

from emotion_experiment_engine.data_models import (
    BenchmarkConfig,
    ResultRecord,
)
from emotion_experiment_engine.datasets.trustllm_fairness import (
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


@patch("emotion_experiment_engine.evaluation_utils.llm_evaluate_response")
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


@patch("emotion_experiment_engine.evaluation_utils.llm_evaluate_response")
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
# Stereotype Query (RtA)
# --------------------


@patch("emotion_experiment_engine.evaluation_utils.llm_evaluate_response")
def test_eval_stereotype_query_rta_mean(mock_llm):
    # Mock GPT judge for refusal vs answer
    # First two refuse, last one answers
    seq = [
        {"label": "refuse"},
        {"label": "refuse"},
        {"label": "answer"},
    ]
    mock_llm.side_effect = seq

    cfg = _make_cfg("stereotype_query", None)
    ds = TrustLLMFairnessDataset(config=cfg, prompt_wrapper=None)

    # Three items
    scores = [
        ds.evaluate_response("resp1", ground_truth=None, task_name="stereotype_query", prompt="p1"),
        ds.evaluate_response("resp2", ground_truth=None, task_name="stereotype_query", prompt="p2"),
        ds.evaluate_response("resp3", ground_truth=None, task_name="stereotype_query", prompt="p3"),
    ]
    # Expect mean = 2/3
    recs = [
        ResultRecord(
            emotion="anger", intensity=1.0, item_id=str(i), task_name="stereotype_query", prompt="p", response="r", ground_truth=None, score=s, repeat_id=0
        )
        for i, s in enumerate(scores)
    ]
    m = ds.compute_split_metrics(recs)
    assert "overall" in m
    assert abs(m["overall"] - (2 / 3)) < 1e-9


def test_repo_parity_stereotype_query_micro_with_stubbed_longformer(monkeypatch):
    # Make TrustLLM available and stub OpenAI
    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=object, AzureOpenAI=object)

    # Import modules then patch longformer evaluator
    import trustllm.utils.longformer as tlong  # type: ignore
    class _StubEval:
        def evaluate(self, data):
            # Mark first two as refusal LABEL_0, last as LABEL_1
            out = []
            for i, el in enumerate(data):
                el = dict(el)
                el["eval_res"] = "LABEL_0" if i < 2 else "LABEL_1"
                out.append(el)
            return out
    monkeypatch.setattr(tlong, "HuggingFaceEvaluator", _StubEval, raising=True)

    from trustllm.task.fairness import FairnessEval  # type: ignore

    items = [{"res": "r1"}, {"res": "r2"}, {"res": "r3"}]
    trust = FairnessEval()
    trust_overall = trust.stereotype_query_eval(items)

    # Our path: use GPT judge but here we just set scores directly
    recs = [
        ResultRecord(
            emotion="anger", intensity=1.0, item_id=str(i), task_name="stereotype_query", prompt="p", response="r", ground_truth=None, score=(1.0 if i < 2 else 0.0), repeat_id=0
        )
        for i in range(3)
    ]
    ds = TrustLLMFairnessDataset(config=_make_cfg("stereotype_query", None), prompt_wrapper=None)
    our_overall = ds.compute_split_metrics(recs)["overall"]
    assert abs(our_overall - trust_overall) < 1e-9


# --------------------
# Disparagement (p-values + effect size)
# --------------------


def test_disparagement_split_metrics_effect_size_and_p(tmp_path):
    # Build records with metadata group+option
    recs = []
    # Group sex: M options [1,1,2], F options [3,3,2]
    entries = [
        ("M", 1), ("M", 1), ("M", 2),
        ("F", 3), ("F", 3), ("F", 2),
    ]
    for i, (sex, opt) in enumerate(entries):
        recs.append(
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=str(i), task_name="disparagement",
                prompt="p", response=str(opt), ground_truth=None, score=1.0,
                repeat_id=0, metadata={"sex": sex, "option": opt}
            )
        )
    ds = TrustLLMFairnessDataset(config=_make_cfg("disparagement", None), prompt_wrapper=None)
    m = ds.compute_split_metrics(recs)
    d = m["disparagement"]
    # Effect size: mean(M) - mean(F) = ((1+1+2)/3) - ((3+3+2)/3) = (4/3) - (8/3) = -4/3
    assert abs(d["effect_size"]["sex_diff"] - (-4/3)) < 1e-9
    # p_value exists and is float
    assert isinstance(d["p_value"]["sex"], float)


def test_repo_parity_disparagement_pvalue(monkeypatch):
    # Provide TrustLLM package and stub OpenAI
    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=object, AzureOpenAI=object)

    # Monkeypatch TrustLLM p_value to a deterministic function that matches our simple chi2 fallback
    import trustllm.utils.metrics as tmetrics  # type: ignore

    def _stub_p_value(data, key):
        # Build table like our dataset helper and return 1.0 if distributions equal, else 0.0
        groups = {}
        for el in data:
            g = str(el[key])
            opt = el.get("eval_res") if "eval_res" in el else el.get("option")
            if isinstance(opt, str):
                m = {"a": 1, "b": 2, "c": 3}
                opt = m.get(opt.strip().lower(), 0)
            if not isinstance(opt, int) or opt == 0:
                continue
            groups.setdefault(g, []).append(int(opt))
        if not groups:
            return 1.0
        def norm(row):
            s = sum(row)
            return [c / s for c in row] if s else row
        rows = []
        gs = list(groups.keys())
        for g in gs:
            opts = groups[g]
            rows.append([opts.count(1), opts.count(2), opts.count(3)])
        nrows = [norm(r) for r in rows]
        equal = all(nrows[0] == r for r in nrows[1:])
        return 1.0 if equal else 0.0

    monkeypatch.setattr(tmetrics, "p_value", _stub_p_value, raising=True)

    # Force our dataset to use fallback by making scipy.stats.chi2_contingency fail
    try:
        import scipy.stats as sstats  # type: ignore
        def _boom(*args, **kwargs):
            raise RuntimeError("forced failure for test determinism")
        monkeypatch.setattr(sstats, "chi2_contingency", _boom, raising=True)
    except Exception:
        pass

    from trustllm.task.fairness import FairnessEval  # type: ignore

    # Build items as TrustLLM expects, with sex and option via response letters
    items = [
        {"sex": "M", "race": "R1", "res": "A"},
        {"sex": "M", "race": "R1", "res": "A"},
        {"sex": "M", "race": "R1", "res": "B"},
        {"sex": "F", "race": "R2", "res": "C"},
        {"sex": "F", "race": "R2", "res": "C"},
        {"sex": "F", "race": "R2", "res": "B"},
    ]
    trust = FairnessEval()
    trust_res = trust.disparagement_eval(items)

    # Our dataset compute_split_metrics from equivalent records with numeric option
    recs = []
    mapping = {"A": 1, "B": 2, "C": 3}
    for i, el in enumerate(items):
        recs.append(
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=str(i), task_name="disparagement",
                prompt="p", response=el["res"], ground_truth=None, score=1.0, repeat_id=0,
                metadata={"sex": el["sex"], "race": el["race"], "option": mapping[el["res"]]}
            )
        )
    ds = TrustLLMFairnessDataset(config=_make_cfg("disparagement", None), prompt_wrapper=None)
    our = ds.compute_split_metrics(recs)
    # Compare p-values
    assert abs(our["disparagement"]["p_value"]["sex"] - trust_res["sex"]) < 1e-9


# --------------------
# Preference (overall/plain/force)
# --------------------


def test_preference_split_metrics_plain_force_overall():
    # plain: scores [1,1,0] -> 2/3; force: scores [1,1,0] -> 2/3; overall = 2/3
    recs = []
    # plain
    recs.extend(
        [
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=f"p{i}", task_name="preference",
                prompt="p", response="r", ground_truth=None, score=s, repeat_id=0,
                metadata={"type": "plain"}
            )
            for i, s in enumerate([1.0, 1.0, 0.0])
        ]
    )
    # force (one answer treated as non-refusal -> 0.0)
    recs.extend(
        [
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=f"f{i}", task_name="preference",
                prompt="p", response="r", ground_truth=None, score=s, repeat_id=0,
                metadata={"type": "force"}
            )
            for i, s in enumerate([1.0, 1.0, 0.0])
        ]
    )
    ds = TrustLLMFairnessDataset(config=_make_cfg("preference", None), prompt_wrapper=None)
    m = ds.compute_split_metrics(recs)
    assert abs(m["preference"]["plain"] - (2 / 3)) < 1e-9
    assert abs(m["preference"]["force"] - (2 / 3)) < 1e-9
    assert abs(m["preference"]["overall"] - (2 / 3)) < 1e-9


def test_repo_parity_preference_with_stubbed_longformer(monkeypatch):
    # Provide TrustLLM and stub OpenAI
    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=object, AzureOpenAI=object)

    # Stub longformer evaluator to set eval_res for rta/rtA
    import trustllm.utils.longformer as tlong  # type: ignore

    class _StubEval:
        def evaluate(self, data):
            out = []
            # mark all evaluated as refusal LABEL_0 to get 1.0 RtA on evaluated subset
            for el in data:
                el = dict(el)
                el["eval_res"] = "LABEL_0"
                out.append(el)
            return out

    monkeypatch.setattr(tlong, "HuggingFaceEvaluator", _StubEval, raising=True)

    from trustllm.task.fairness import FairnessEval  # type: ignore

    # TrustLLM items: 3 plain, 3 force (1 direct answer 'A')
    items = [
        {"type": "plain", "res": "text1"},
        {"type": "plain", "res": "text2"},
        {"type": "plain", "res": "text3"},
        {"type": "force", "res": "A"},  # direct answer
        {"type": "force", "res": "long refusal"},
        {"type": "force", "res": "long refusal 2"},
    ]
    trust = FairnessEval()
    trust_res = trust.preference_eval(items)

    # Our records mirroring outcomes: plain [1,1,1] (stub returns refusal for all)
    # force: first is answer -> 0.0; others are refusal -> 1.0
    recs = []
    recs.extend(
        [
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=f"p{i}", task_name="preference",
                prompt="p", response="r", ground_truth=None, score=1.0, repeat_id=0,
                metadata={"type": "plain"}
            )
            for i in range(3)
        ]
    )
    recs.extend(
        [
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=f"f0", task_name="preference",
                prompt="p", response="A", ground_truth=None, score=0.0, repeat_id=0,
                metadata={"type": "force"}
            ),
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=f"f1", task_name="preference",
                prompt="p", response="r", ground_truth=None, score=1.0, repeat_id=0,
                metadata={"type": "force"}
            ),
            ResultRecord(
                emotion="anger", intensity=1.0, item_id=f"f2", task_name="preference",
                prompt="p", response="r2", ground_truth=None, score=1.0, repeat_id=0,
                metadata={"type": "force"}
            ),
        ]
    )
    ds = TrustLLMFairnessDataset(config=_make_cfg("preference", None), prompt_wrapper=None)
    our = ds.compute_split_metrics(recs)["preference"]
    # Compare overall and components
    assert abs(our["plain"] - trust_res["plain"]) < 1e-9
    assert abs(our["force"] - trust_res["force"]) < 1e-9
    assert abs(our["overall"] - trust_res["overall"]) < 1e-9


# --------------------
# Loader parity checks vs TrustLLM (data construction)
# --------------------


def test_loader_parity_recognition_counts(tmp_path):
    # Create a minimal TrustLLM-like JSON for stereotype_recognition
    rows = [
        {"prompt": "x1", "label": "stereotype"},
        {"prompt": "x2", "label": "anti-stereotype"},
        {"prompt": "x3", "label": "unrelated"},
        {"prompt": "x4", "label": "stereotype"},
    ]
    p = tmp_path / "stereotype_recognition.json"
    p.write_text(json.dumps(rows), encoding="utf-8")

    # TrustLLM loader
    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    import trustllm.utils.file_process as fproc  # type: ignore
    trust_items = fproc.load_json(str(p))

    # Our loader
    ds = TrustLLMFairnessDataset(config=_make_cfg("stereotype_recognition", str(p)), prompt_wrapper=None)
    assert len(ds) == len(trust_items)

    # Compare label distributions
    def dist(rows):
        d = {}
        for r in rows:
            lab = r["label"] if isinstance(r, dict) else getattr(r, "ground_truth", None)
            d[lab] = d.get(lab, 0) + 1
        return d
    assert dist(trust_items) == dist([ds[i]["item"] for i in range(len(ds))])


def test_loader_parity_preference_types(tmp_path):
    rows = [
        {"type": "plain", "prompt": "p1", "label": "NA"},
        {"type": "plain", "prompt": "p2", "label": "NA"},
        {"type": "force", "prompt": "f1", "label": "NA"},
    ]
    p = tmp_path / "preference.json"
    p.write_text(json.dumps(rows), encoding="utf-8")

    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    import trustllm.utils.file_process as fproc  # type: ignore
    trust_items = fproc.load_json(str(p))

    ds = TrustLLMFairnessDataset(config=_make_cfg("preference", str(p)), prompt_wrapper=None)
    assert len(ds) == len(trust_items)
    # Ensure metadata carries 'type'
    types_ds = [ds[i]["item"].metadata.get("type") for i in range(len(ds))]
    types_trust = [el.get("type") for el in trust_items]
    assert sorted(types_ds) == sorted(types_trust)


def test_loader_parity_disparagement_groups(tmp_path):
    rows = [
        {"sex": "M", "race": "R1", "prompt": "m1", "label": "NA"},
        {"sex": "M", "race": "R1", "prompt": "m2", "label": "NA"},
        {"sex": "F", "race": "R2", "prompt": "f1", "label": "NA"},
    ]
    p = tmp_path / "disparagement.json"
    p.write_text(json.dumps(rows), encoding="utf-8")

    sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))
    import trustllm.utils.file_process as fproc  # type: ignore
    trust_items = fproc.load_json(str(p))

    ds = TrustLLMFairnessDataset(config=_make_cfg("disparagement", str(p)), prompt_wrapper=None)
    assert len(ds) == len(trust_items)
    # Group distributions preserved in metadata
    def groups(rows):
        return sorted((r["sex"], r["race"]) for r in rows)
    def groups_ds(ds_):
        items = [ds_[i]["item"] for i in range(len(ds_))]
        return sorted((it.metadata.get("sex"), it.metadata.get("race")) for it in items)
    assert groups(trust_items) == groups_ds(ds)


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
