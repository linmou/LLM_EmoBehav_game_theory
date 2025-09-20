"""
Test responsible for TrustLLMEthicsDataset implicit_ethics macro parity.
Purpose: ensure macro excludes 'not sure' when present, matching TrustLLM.
Uses a small JSON fixture with expected counts/per-class/macro.
"""

import json
from pathlib import Path

from emotion_memory_experiments.data_models import BenchmarkConfig, ResultRecord
from emotion_memory_experiments.datasets.trustllm_ethics import TrustLLMEthicsDataset


def test_implicit_ethics_parity_fixture():
    fixture_path = Path("emotion_memory_experiments/tests/test_data/trustllm/implicit_ethics_parity.json")
    data = json.loads(fixture_path.read_text())

    # Build records
    recs = []
    for i, it in enumerate(data["items"]):
        recs.append(
            ResultRecord(
                emotion="anger",
                intensity=1.0,
                item_id=f"{i}",
                task_name="implicit_ethics",
                prompt="p",
                response="r",
                ground_truth=it["gt"],
                score=1.0 if it["correct"] else 0.0,
                repeat_id=0,
            )
        )

    cfg = BenchmarkConfig(
        name="trustllm_ethics",
        task_type="implicit_ethics",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )
    ds = TrustLLMEthicsDataset(config=cfg, prompt_wrapper=None)
    m = ds.compute_split_metrics(recs)
    ie = m["implicit_ethics"]

    exp = data["expected"]
    # counts
    assert ie["counts"]["wrong"] == exp["counts"]["wrong"]
    assert ie["counts"]["not wrong"] == exp["counts"]["not wrong"]
    assert ie["counts"]["not sure"] == exp["counts"]["not sure"]
    # per-class
    assert abs(ie["per_class_acc"]["wrong"] - exp["per_class_acc"]["wrong"]) < 1e-6
    assert abs(ie["per_class_acc"]["not wrong"] - exp["per_class_acc"]["not wrong"]) < 1e-6
    assert abs(ie["per_class_acc"]["not sure"] - exp["per_class_acc"]["not sure"]) < 1e-6
    # macro excludes not sure
    assert abs(ie["macro_acc"] - exp["macro_acc_excluding_not_sure"]) < 1e-6

