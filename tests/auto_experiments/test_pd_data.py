"""Tests: auto_experiments/task-similarity/pd_data.py
Purpose: validate PD data loading/splitting and label construction."""

from pathlib import Path

from auto_experiments.task_similarity import pd_data
from auto_experiments.task_similarity.pd_prompt_builder import PromptPair, PairMeta


class DummyPair(PromptPair):
    def __init__(self, idx: int):
        meta = PairMeta(
            opt_a=f"a{idx}",
            opt_b=f"b{idx}",
            defect_label="A",
            cooperate_label="B",
            description=f"desc{idx}",
        )
        super().__init__(positive=f"pos{idx}", negative=f"neg{idx}", meta=meta)


def test_split_pairs_reproducible():
    pairs = [DummyPair(i) for i in range(6)]
    train1, test1 = pd_data.split_pairs(pairs, train_ratio=0.5, seed=42)
    train2, test2 = pd_data.split_pairs(pairs, train_ratio=0.5, seed=42)
    assert [p.positive for p in train1] == [p.positive for p in train2]
    assert [p.positive for p in test1] == [p.positive for p in test2]
    assert len(train1) == 3
    assert len(test1) == 3


def test_build_repreader_dataset_labels():
    pairs = [DummyPair(0), DummyPair(1)]
    ds = pd_data.build_repreader_dataset(pairs)
    assert ds["data"] == ["pos0", "neg0", "pos1", "neg1"]
    assert ds["labels"] == [[1, 0], [1, 0]]
