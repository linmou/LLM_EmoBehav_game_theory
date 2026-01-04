"""
Responsible: auto_experiments/task_similarity/pd_probe_transfer.py
Purpose: Train per-layer PD probes (linear SVM) on (delta_pd vs delta_pd_cooperate)
         and evaluate transfer ROC-AUC on delta_emotion vs chosen_behavior.

This module is intentionally minimal: core math is unit-tested, and filesystem/CLI
plumbing (if needed) can live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class LinearProbe:
    w: np.ndarray  # (d,)
    b: float

    def score(self, x: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x, dtype=np.float32)
        if x2.ndim != 2:
            raise ValueError(f"x must be 2D (n,d), got {x2.shape}")
        if x2.shape[1] != self.w.shape[0]:
            raise ValueError(f"dim mismatch: x has d={x2.shape[1]} but w has d={self.w.shape[0]}")
        return (x2 @ self.w.astype(np.float32, copy=False)) + np.float32(self.b)


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x2 = np.asarray(x, dtype=np.float32)
    if x2.ndim != 2:
        raise ValueError(f"x must be 2D, got {x2.shape}")
    n = np.linalg.norm(x2, axis=1, keepdims=True)
    return x2 / (n + np.float32(eps))


def train_pd_probes_per_layer(
    *,
    delta_pd: np.ndarray,
    delta_pd_cooperate: np.ndarray,
    train_item_ids: Sequence[int],
    l2_normalize: bool,
) -> List[LinearProbe]:
    """
    Train one linear SVM per layer to separate delta_pd (label=1) vs delta_pd_cooperate (label=0).

    Args:
        delta_pd: shape (n_int, n_items, n_layers, d)
        delta_pd_cooperate: same shape
    """
    a = np.asarray(delta_pd, dtype=np.float32)
    b = np.asarray(delta_pd_cooperate, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 4:
        raise ValueError(f"expected 4D (n_int,n_items,n_layers,d), got {a.shape}")

    n_int, n_items, n_layers, d = a.shape
    train = [int(i) for i in train_item_ids]
    if any(i < 0 or i >= n_items for i in train):
        raise ValueError("train_item_ids out of range")
    if len(train) == 0:
        raise ValueError("train_item_ids empty")

    from sklearn.svm import LinearSVC

    probes: List[LinearProbe] = []
    for layer in range(n_layers):
        x_pos = a[:, train, layer, :].reshape(-1, d)
        x_neg = b[:, train, layer, :].reshape(-1, d)
        x = np.concatenate([x_pos, x_neg], axis=0)
        y = np.concatenate(
            [np.ones((x_pos.shape[0],), dtype=np.int64), np.zeros((x_neg.shape[0],), dtype=np.int64)],
            axis=0,
        )
        if l2_normalize:
            x = _l2_normalize_rows(x)

        clf = LinearSVC(C=1.0, dual=False, random_state=0, max_iter=2000)
        clf.fit(x, y)

        w = np.asarray(clf.coef_.reshape(-1), dtype=np.float32)
        bb = float(np.asarray(clf.intercept_, dtype=np.float32).reshape(-1)[0])
        probe = LinearProbe(w=w, b=bb)

        # Orient so that PD-defect deltas have higher score than PD-cooperate deltas.
        s_pos = probe.score(x_pos if not l2_normalize else _l2_normalize_rows(x_pos))
        s_neg = probe.score(x_neg if not l2_normalize else _l2_normalize_rows(x_neg))
        if float(np.mean(s_pos)) <= float(np.mean(s_neg)):
            probe = LinearProbe(w=-probe.w, b=-probe.b)
        probes.append(probe)

    return probes


def evaluate_transfer_auc_by_layer(
    *,
    probes: Sequence[LinearProbe],
    delta_emotion: np.ndarray,
    test_item_ids: Sequence[int],
    y_defect: np.ndarray,
    intensity_index: int,
    l2_normalize: bool,
) -> np.ndarray:
    """
    Evaluate per-layer transfer ROC-AUC: probe score on delta_emotion predicts defect label.
    Returns shape (n_layers,), NaN when AUC is undefined (only one class present).
    """
    de = np.asarray(delta_emotion, dtype=np.float32)
    if de.ndim != 4:
        raise ValueError(f"delta_emotion must be 4D (n_int,n_items,n_layers,d), got {de.shape}")
    n_int, n_items, n_layers, d = de.shape
    if len(probes) != n_layers:
        raise ValueError(f"probes length mismatch: {len(probes)} vs n_layers={n_layers}")
    if intensity_index < 0 or intensity_index >= n_int:
        raise ValueError("intensity_index out of range")

    test = [int(i) for i in test_item_ids]
    if any(i < 0 or i >= n_items for i in test):
        raise ValueError("test_item_ids out of range")
    y = np.asarray(y_defect, dtype=np.int64)
    if y.shape != (len(test),):
        raise ValueError(f"y_defect shape mismatch: expected {(len(test),)} got {y.shape}")

    from sklearn.metrics import roc_auc_score

    out = np.full((n_layers,), np.nan, dtype=np.float32)
    if len(np.unique(y)) < 2:
        return out

    for layer in range(n_layers):
        x = de[intensity_index, test, layer, :].reshape(len(test), d)
        if l2_normalize:
            x = _l2_normalize_rows(x)
        s = probes[layer].score(x)
        try:
            out[layer] = float(roc_auc_score(y, s))
        except ValueError:
            out[layer] = np.nan
    return out

