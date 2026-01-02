"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Ensure the progress wrapper is safe even when tqdm is unavailable.
"""

import builtins


def test_progress_falls_back_without_tqdm(monkeypatch):
    from auto_experiments.task_similarity import emotion_pd_delta_similarity as mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("tqdm intentionally missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    xs = [1, 2, 3]
    got = list(mod.progress(xs, total=len(xs), desc="x"))
    assert got == xs
