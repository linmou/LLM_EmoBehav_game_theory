"""
Tests for auto_experiments/task_similarity/pd_data.py.

Focus: pair splitting and repreader-compatible dataset construction.
"""

from ..pd_data import build_repreader_dataset, split_pairs
from ..pd_prompt_builder import PromptPair, PairMeta


def _make_pair(idx: int) -> PromptPair:
    meta = PairMeta(
        opt_a=f"opt_a_{idx}",
        opt_b=f"opt_b_{idx}",
        defect_label="A",
        cooperate_label="B",
        description=f"desc_{idx}",
    )
    return PromptPair(
        positive=f"pos_{idx}",
        negative=f"neg_{idx}",
        meta=meta,
    )


def test_build_repreader_dataset_structure():
    pairs = [_make_pair(i) for i in range(3)]

    ds = build_repreader_dataset(pairs)
    data = ds["data"]
    labels = ds["labels"]

    # 2 entries (pos, neg) per pair
    assert len(data) == 2 * len(pairs)
    assert data[0] == pairs[0].positive
    assert data[1] == pairs[0].negative

    # Labels are [1, 0] per pair (defection as positive)
    assert len(labels) == len(pairs)
    assert all(lbl == [1, 0] for lbl in labels)


def test_split_pairs_is_deterministic_and_partitioned():
    pairs = [_make_pair(i) for i in range(10)]

    train1, test1 = split_pairs(pairs, train_ratio=0.6, seed=42)
    train2, test2 = split_pairs(pairs, train_ratio=0.6, seed=42)

    # Deterministic for fixed seed
    assert {p.meta.description for p in train1} == {
        p.meta.description for p in train2
    }
    assert {p.meta.description for p in test1} == {
        p.meta.description for p in test2
    }

    # Partition: no overlap, union equals original
    train_set = {p.meta.description for p in train1}
    test_set = {p.meta.description for p in test1}
    all_set = {p.meta.description for p in pairs}

    assert train_set.isdisjoint(test_set)
    assert train_set.union(test_set) == all_set
