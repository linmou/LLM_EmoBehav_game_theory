"""
Responsible: auto_experiments/task_similarity/pd_probe_transfer.py
Purpose: Unit tests for per-layer PD probe training and transfer AUC evaluation.
"""

from __future__ import annotations

import numpy as np


def test_train_pd_probes_scores_pd_correct_direction() -> None:
    from auto_experiments.task_similarity.pd_probe_transfer import train_pd_probes_per_layer

    rng = np.random.default_rng(0)
    n_int, n_items, n_layers, d = 2, 10, 3, 4
    train_items = list(range(6))

    e1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    delta_pd = rng.normal(scale=0.01, size=(n_int, n_items, n_layers, d)).astype(np.float32)
    delta_pd += e1  # defect-like
    delta_pd_coop = rng.normal(scale=0.01, size=(n_int, n_items, n_layers, d)).astype(np.float32)
    delta_pd_coop -= e1  # cooperate-like
    # Add asymmetry in post-control layers to mimic non-linear propagation.
    delta_pd_coop[:, :, 2, :] += np.array([0.0, 0.2, 0.0, 0.0], dtype=np.float32)

    probes = train_pd_probes_per_layer(
        delta_pd=delta_pd,
        delta_pd_cooperate=delta_pd_coop,
        train_item_ids=train_items,
        l2_normalize=True,
    )
    assert len(probes) == n_layers

    # The oriented score should rank PD deltas higher than PD_cooperate deltas on train items.
    for layer, pr in enumerate(probes):
        s_pos = pr.score(delta_pd[:, train_items, layer, :].reshape(-1, d))
        s_neg = pr.score(delta_pd_coop[:, train_items, layer, :].reshape(-1, d))
        assert float(np.mean(s_pos)) > float(np.mean(s_neg))


def test_transfer_auc_high_when_emotion_delta_encodes_behavior() -> None:
    from auto_experiments.task_similarity.pd_probe_transfer import (
        evaluate_transfer_auc_by_layer,
        train_pd_probes_per_layer,
    )

    rng = np.random.default_rng(1)
    n_int, n_items, n_layers, d = 2, 12, 4, 5
    train_items = list(range(7))
    test_items = list(range(7, n_items))
    y = np.array([0, 1, 0, 1, 0], dtype=np.int64)  # labels for test items

    e1 = np.zeros((d,), dtype=np.float32)
    e1[0] = 1.0

    delta_pd = rng.normal(scale=0.02, size=(n_int, n_items, n_layers, d)).astype(np.float32) + e1
    delta_pd_coop = rng.normal(scale=0.02, size=(n_int, n_items, n_layers, d)).astype(np.float32) - e1

    probes = train_pd_probes_per_layer(
        delta_pd=delta_pd,
        delta_pd_cooperate=delta_pd_coop,
        train_item_ids=train_items,
        l2_normalize=True,
    )

    # Emotion deltas on test items: layer 1 carries the label signal, other layers are noise.
    delta_emotion = rng.normal(scale=0.05, size=(n_int, n_items, n_layers, d)).astype(np.float32)
    for i, item_id in enumerate(test_items):
        delta_emotion[:, item_id, 1, :] += (1.0 if y[i] == 1 else -1.0) * e1

    aucs = evaluate_transfer_auc_by_layer(
        probes=probes,
        delta_emotion=delta_emotion,
        test_item_ids=test_items,
        y_defect=y,
        intensity_index=0,
        l2_normalize=True,
    )
    assert aucs.shape == (n_layers,)
    assert float(aucs[1]) > 0.9


def test_transfer_auc_nan_when_only_one_class_present() -> None:
    from auto_experiments.task_similarity.pd_probe_transfer import (
        evaluate_transfer_auc_by_layer,
        train_pd_probes_per_layer,
    )

    rng = np.random.default_rng(2)
    n_int, n_items, n_layers, d = 1, 8, 2, 3
    train_items = list(range(5))
    test_items = list(range(5, n_items))

    delta_pd = rng.normal(size=(n_int, n_items, n_layers, d)).astype(np.float32)
    delta_pd_coop = rng.normal(size=(n_int, n_items, n_layers, d)).astype(np.float32)
    probes = train_pd_probes_per_layer(
        delta_pd=delta_pd,
        delta_pd_cooperate=delta_pd_coop,
        train_item_ids=train_items,
        l2_normalize=True,
    )

    delta_emotion = rng.normal(size=(n_int, n_items, n_layers, d)).astype(np.float32)
    y = np.ones((len(test_items),), dtype=np.int64)  # all defect
    aucs = evaluate_transfer_auc_by_layer(
        probes=probes,
        delta_emotion=delta_emotion,
        test_item_ids=test_items,
        y_defect=y,
        intensity_index=0,
        l2_normalize=True,
    )
    assert np.isnan(aucs).all()

