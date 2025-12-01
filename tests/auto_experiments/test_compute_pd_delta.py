"""Tests: auto_experiments/task-similarity/compute_pd_delta.py
Purpose: layer selection and validation guardrails."""

import pytest
import numpy as np

from delta_activation_engine.backends.hf import select_middle_third_layers
from auto_experiments.task_similarity import compute_pd_delta


def test_select_middle_third_layers_imported():
    assert select_middle_third_layers(12) == [4, 5, 6, 7]
    assert select_middle_third_layers(3) == [1]


def test_requires_layer_or_middle_third():
    with pytest.raises(ValueError):
        compute_pd_delta.resolve_control_layers(num_layers=12, layer=None, use_middle_third=False)
    assert compute_pd_delta.resolve_control_layers(12, layer=5, use_middle_third=False) == [5]
    assert compute_pd_delta.resolve_control_layers(12, layer=None, use_middle_third=True) == [4, 5, 6, 7]


def test_run_delta_loads_layer_vector_directory(tmp_path, monkeypatch):
    vector_dir = tmp_path / "results" / "Qwen2.5-0.5B-Instruct" / "layer_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    layer0 = np.array([0.1, 0.2], dtype=np.float32)
    layer1 = np.array([1.0, 1.1], dtype=np.float32)
    np.save(vector_dir / "layer_0.npy", layer0)
    np.save(vector_dir / "layer_1.npy", layer1)

    monkeypatch.setattr(compute_pd_delta, "get_generic_probes", lambda: ["p1", "p2"])

    class _DummyLayer:
        pass

    class _DummyModel:
        def __init__(self):
            class Cfg:
                def __init__(self):
                    self.num_hidden_layers = 2

            class Inner:
                def __init__(self):
                    self.layers = [_DummyLayer(), _DummyLayer()]

            self.config = Cfg()
            self.model = Inner()

    monkeypatch.setattr(
        compute_pd_delta,
        "AutoModelForCausalLM",
        type("F", (), {"from_pretrained": staticmethod(lambda *_a, **_k: _DummyModel())}),
    )
    monkeypatch.setattr(
        compute_pd_delta,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(lambda *_a, **_k: object())}),
    )

    call_counter = {"n": 0}

    def _fake_collect_hidden(_model, _tokenizer, _prompts, measurement_layer, *_args, **_kwargs):
        call_counter["n"] += 1
        base = np.zeros(2, dtype=np.float32)
        steered = np.ones(2, dtype=np.float32)
        vec = base if call_counter["n"] == 1 else steered
        return vec

    monkeypatch.setattr(compute_pd_delta, "_collect_final_token_hidden", _fake_collect_hidden)

    hook_calls = []

    def _fake_register(layer_module, vec, intensity):
        hook_calls.append((layer_module, vec.copy(), intensity))

        class H:
            def remove(self):
                pass

        return H()

    monkeypatch.setattr(compute_pd_delta, "_register_control_hook", _fake_register)
    monkeypatch.setattr(
        compute_pd_delta,
        "resolve_control_layers",
        lambda num_layers, layer, use_middle_third: [0, 1],
    )

    result = compute_pd_delta.run_delta(
        model_path="dummy",
        vector_path=vector_dir,
        layer=None,
        use_middle_third=True,
        intensity=1.0,
        output_dir=tmp_path / "out",
        max_length=8,
        batch_size=2,
        seed=0,
    )

    assert result["measurement_layer"] == 1
    assert len(hook_calls) == 2
    np.testing.assert_allclose(hook_calls[0][1], layer0)
    np.testing.assert_allclose(hook_calls[1][1], layer1)
    assert all(call[-1] == 1.0 for call in hook_calls)


def test_run_delta_passes_torch_dtype(tmp_path, monkeypatch):
    vec_path = tmp_path / "vec.npy"
    np.save(vec_path, np.ones(2, dtype=np.float32))

    monkeypatch.setattr(compute_pd_delta, "get_generic_probes", lambda: ["p1"])

    class _DummyLayer:
        pass

    class _DummyModel:
        def __init__(self):
            class Cfg:
                def __init__(self):
                    self.num_hidden_layers = 1

            class Inner:
                def __init__(self):
                    self.layers = [_DummyLayer()]

            self.config = Cfg()
            self.model = Inner()

    call_kwargs = {}

    def _fake_from_pretrained(*_args, **kwargs):
        call_kwargs.update(kwargs)
        return _DummyModel()

    monkeypatch.setattr(
        compute_pd_delta,
        "AutoModelForCausalLM",
        type("F", (), {"from_pretrained": staticmethod(_fake_from_pretrained)}),
    )
    monkeypatch.setattr(
        compute_pd_delta,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(lambda *_a, **_k: object())}),
    )

    monkeypatch.setattr(
        compute_pd_delta,
        "_collect_final_token_hidden",
        lambda *_a, **_k: np.zeros(2, dtype=np.float32),
    )
    monkeypatch.setattr(
        compute_pd_delta,
        "_register_control_hook",
        lambda *_a, **_k: type("H", (), {"remove": lambda _self: None})(),
    )

    compute_pd_delta.run_delta(
        model_path="dummy",
        vector_path=vec_path,
        layer=0,
        use_middle_third=False,
        intensity=1.0,
        output_dir=tmp_path / "out",
        max_length=8,
        batch_size=2,
        seed=0,
    )

    assert "torch_dtype" in call_kwargs
    assert "dtype" not in call_kwargs
