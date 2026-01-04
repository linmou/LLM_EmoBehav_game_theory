#!/usr/bin/env python3
"""
Responsible file: emotion_experiment_engine/benchmark_component_registry.py
Purpose: Ensure game_theory_decision is wired as a registry alias for game_theory.
"""

from __future__ import annotations

from emotion_experiment_engine.benchmark_component_registry import BENCHMARK_SPECS
from emotion_experiment_engine.datasets.games import GameTheoryDataset


def test_game_theory_decision_alias_exists() -> None:
    spec = BENCHMARK_SPECS[("game_theory_decision", "*")]
    assert spec.dataset_class is GameTheoryDataset

