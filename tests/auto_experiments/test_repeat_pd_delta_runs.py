"""Tests: auto_experiments/task-similarity/repeat_pd_delta_runs.py
Purpose: batch runner should call run_delta with sequential seeds and forward arguments."""

from pathlib import Path

from auto_experiments.task_similarity import repeat_pd_delta_runs as mod


def test_run_batch_calls_run_delta_with_incrementing_seeds(tmp_path, monkeypatch):
    calls = []

    def _fake_run_delta(**kwargs):
        calls.append(kwargs)
        return {"seed": kwargs["seed"], "tag": "ok"}

    monkeypatch.setattr(mod.compute_pd_delta, "run_delta", _fake_run_delta)

    results = mod.run_batch(
        model_path="dummy-model",
        vector_path=tmp_path / "layer_vectors",
        output_dir=tmp_path / "out",
        layer=3,
        middle_third=False,
        intensity=1.2,
        max_length=64,
        batch_size=4,
        start_seed=7,
        num_runs=3,
    )

    assert [c["seed"] for c in calls] == [7, 8, 9]
    for call in calls:
        assert call["layer"] == 3
        assert call["use_middle_third"] is False
        assert call["intensity"] == 1.2
        assert call["model_path"] == "dummy-model"
        assert call["vector_path"] == tmp_path / "layer_vectors"
    assert [r["seed"] for r in results] == [7, 8, 9]


def test_run_batch_logs_progress(tmp_path, monkeypatch, capsys):
    def _fake_run_delta(**kwargs):
        return {"seed": kwargs["seed"]}

    monkeypatch.setattr(mod.compute_pd_delta, "run_delta", _fake_run_delta)

    mod.run_batch(
        model_path="dummy-model",
        vector_path=tmp_path / "layer_vectors",
        output_dir=tmp_path / "out",
        layer=None,
        middle_third=True,
        intensity=1.0,
        max_length=16,
        batch_size=2,
        start_seed=2,
        num_runs=2,
    )

    out = capsys.readouterr().out
    assert "[1/2] seed=2 start" in out
    assert "[1/2] seed=2 done" in out
    assert "[2/2] seed=3 start" in out
    assert "[2/2] seed=3 done" in out
    assert "completed 2 runs starting at seed 2" in out
