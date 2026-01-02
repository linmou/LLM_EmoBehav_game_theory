"""Tests: auto_experiments/task-similarity/delta_for_steering_vectors.py
Purpose: Ensure batch delta runner walks steering vector tree and calls compute_pd_delta.run_delta correctly."""

from auto_experiments.task_similarity import delta_for_steering_vectors as mod


def test_run_for_steering_vectors_calls_run_delta_for_each_seed(tmp_path, monkeypatch):
    # Gherkin-style behavior description:
    # Given a steering root with timestamp and seed subdirectories
    # When run_for_steering_vectors is called
    # Then compute_pd_delta.run_delta is invoked once per seed directory
    #  And layer is fixed
    #  And use_middle_third is False
    #  And vector_path points to the per-seed layer_vectors directory
    #  And output_dir nests timestamp/seed under the delta root.
    steering_root = tmp_path / "steering_vectors" / "Qwen2.5-0.5B-Instruct"
    ts1 = steering_root / "20250101_000000" / "seed_0"
    ts2 = steering_root / "20250101_010000" / "seed_5"
    for p in (ts1, ts2):
        (p / "layer_vectors").mkdir(parents=True)

    delta_root = tmp_path / "delta_out"
    calls = []

    def _fake_run_delta(**kwargs):
        calls.append(kwargs)
        return {"seed": kwargs["seed"], "ok": True}

    monkeypatch.setattr(mod.compute_pd_delta, "run_delta", _fake_run_delta)

    results = mod.run_for_steering_vectors(
        model_path="dummy-model",
        steering_root=steering_root,
        delta_root=delta_root,
        layer=13,
        intensity=1.5,
        max_length=128,
        batch_size=4,
    )

    # We should have one call per seed directory
    assert len(calls) == 2
    assert [c["model_path"] for c in calls] == ["dummy-model", "dummy-model"]
    assert [c["layer"] for c in calls] == [13, 13]
    assert all(not c["use_middle_third"] for c in calls)

    # Vector paths should be the per-seed layer_vectors directories
    assert {c["vector_path"] for c in calls} == {ts1 / "layer_vectors", ts2 / "layer_vectors"}

    # Output dirs should be nested under delta_root/<timestamp>/<seed>
    expected_out_dirs = {
        delta_root / "20250101_000000" / "seed_0",
        delta_root / "20250101_010000" / "seed_5",
    }
    assert {c["output_dir"] for c in calls} == expected_out_dirs

    # Seeds should increment with call index (0, 1, ...)
    assert [c["seed"] for c in calls] == [0, 1]
    assert [r["seed"] for r in results] == [0, 1]


def test_run_for_steering_vectors_reads_best_layer_when_missing(tmp_path, monkeypatch):
    # Given a steering root whose parent has layer_metrics.json
    root = tmp_path / "steering_vectors"
    steering_root = root / "Qwen2.5-0.5B-Instruct"
    seed_dir = steering_root / "20250101_000000" / "seed_0"
    (seed_dir / "layer_vectors").mkdir(parents=True)

    layer_metrics = {
        "layer_accuracies": {"9": 0.5, "13": 0.6},
        "best_layer": 13,
        "best_accuracy": 0.6,
    }
    metrics_path = root / "layer_metrics.json"
    metrics_path.write_text(__import__("json").dumps(layer_metrics), encoding="utf-8")

    captured_layers: list[int] = []

    def _fake_run_delta(**kwargs):
        captured_layers.append(kwargs["layer"])
        return {"seed": kwargs["seed"], "ok": True}

    monkeypatch.setattr(mod.compute_pd_delta, "run_delta", _fake_run_delta)

    mod.run_for_steering_vectors(
        model_path="dummy-model",
        steering_root=steering_root,
        delta_root=tmp_path / "delta_out",
        layer=None,
        intensity=1.0,
        max_length=32,
        batch_size=2,
    )

    assert captured_layers == [13]


def test_run_for_steering_vectors_can_use_middle_third(tmp_path, monkeypatch):
    # When use_middle_third is True, we should not require layer_metrics.json
    # and should pass use_middle_third=True and layer=None through to run_delta.
    steering_root = tmp_path / "steering_vectors" / "Qwen2.5-0.5B-Instruct"
    seed_dir = steering_root / "20250101_000000" / "seed_0"
    (seed_dir / "layer_vectors").mkdir(parents=True)

    calls = []

    def _fake_run_delta(**kwargs):
        calls.append(kwargs)
        return {"seed": kwargs["seed"], "ok": True}

    monkeypatch.setattr(mod.compute_pd_delta, "run_delta", _fake_run_delta)

    # No layer_metrics.json created here; this would fail if _resolve_layer were used.
    mod.run_for_steering_vectors(
        model_path="dummy-model",
        steering_root=steering_root,
        delta_root=tmp_path / "delta_out",
        layer=None,
        intensity=1.0,
        max_length=64,
        batch_size=4,
        use_middle_third=True,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["layer"] is None
    assert call["use_middle_third"] is True
    assert call["vector_path"] == seed_dir / "layer_vectors"


def test_main_uses_defaults_and_calls_run_for_steering_vectors(monkeypatch, capsys):
    # Gherkin:
    # Given no CLI arguments
    # When main() is invoked
    # Then run_for_steering_vectors is called once with default params
    #  And middle_third is enabled by default.
    calls = []

    def _fake_run_for_steering_vectors(**kwargs):
        calls.append(kwargs)
        return [{"seed": 0}]

    monkeypatch.setattr(mod, "run_for_steering_vectors", _fake_run_for_steering_vectors)
    monkeypatch.setattr(
        mod, "__name__", "__main__", raising=False  # ensure main guard would trigger if used
    )

    import sys

    argv_backup = sys.argv
    try:
        sys.argv = ["delta_for_steering_vectors.py"]
        mod.main()
    finally:
        sys.argv = argv_backup

    assert len(calls) == 1
    call = calls[0]

    # Defaults should match the convenience Qwen2.5-0.5B config.
    assert "/Qwen2.5-0.5B-Instruct" in call["model_path"]
    assert "steering_vectors/Qwen2.5-0.5B-Instruct" in str(call["steering_root"])
    assert "results/delta/Qwen2.5-0.5B-steering_vectors_midthird" in str(
        call["delta_root"]
    )
    assert call["layer"] is None
    assert call["intensity"] == 1.5
    assert call["max_length"] == 256
    assert call["batch_size"] == 8
    assert call["use_middle_third"] is True


def test_run_for_steering_vectors_uses_tqdm_progress(tmp_path, monkeypatch):
    # Given two seed directories
    steering_root = tmp_path / "steering_vectors" / "Qwen2.5-0.5B-Instruct"
    ts1 = steering_root / "20250101_000000" / "seed_0"
    ts2 = steering_root / "20250101_010000" / "seed_1"
    for p in (ts1, ts2):
        (p / "layer_vectors").mkdir(parents=True)

    # Stub out run_delta so we don't touch models
    monkeypatch.setattr(
        mod.compute_pd_delta,
        "run_delta",
        lambda **kwargs: {"seed": kwargs["seed"]},
    )

    seen: dict[str, object] = {}

    def _fake_tqdm(iterable, *args, **kwargs):
        # Materialize iterable so we can inspect it
        items = list(iterable)
        seen["total"] = kwargs.get("total")
        seen["items"] = items
        return items

    monkeypatch.setattr(mod, "tqdm", _fake_tqdm)

    mod.run_for_steering_vectors(
        model_path="dummy",
        steering_root=steering_root,
        delta_root=tmp_path / "delta_out",
        layer=None,
        intensity=1.0,
        max_length=32,
        batch_size=2,
        use_middle_third=True,
    )

    assert seen["total"] == 2
    assert set(seen["items"]) == {ts1, ts2}
