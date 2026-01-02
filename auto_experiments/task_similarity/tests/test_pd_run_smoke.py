"""
Smoke test for run_pd_defection_experiment.run.

This does NOT load a real model. Instead, it monkeypatches:
- build_pd_pair_bundle: returns a tiny fixed bundle
- train_pd_repreader: returns synthetic accuracies and vectors
- AutoModelForCausalLM / AutoTokenizer / _decision_rate / _register_control_hook:
  replaced by light stubs, so we only validate wiring, not HF internals.

Responsible: auto_experiments/task_similarity/run_pd_defection_experiment.py
Purpose: Ensure the high-level run() API produces a coherent result dict and
         writes metrics/vectors without touching external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .. import pd_data
from ..pd_prompt_builder import PairMeta, PromptPair
from .. import run_pd_defection_experiment as mod


class _DummyLayer:
    def parameters(self):
        # Minimal iterable with one tensor-like object; not actually used because
        # we patch _register_control_hook.
        import torch

        w = torch.zeros(1, 1)
        return [w]


class _DummyModel:
    def __init__(self, num_layers: int = 2):
        class Cfg:
            def __init__(self, n):
                self.num_hidden_layers = n

        class Inner:
            def __init__(self, n):
                self.layers = [_DummyLayer() for _ in range(n)]

        self.config = Cfg(num_layers)
        self.model = Inner(num_layers)


class _DummyTokenizer:
    def __call__(self, *args, **kwargs):
        # Not used in the smoke test; run() will not hit real encoding
        class Enc:
            def __init__(self):
                self.input_ids = []
                self.attention_mask = []

        return Enc()


def _make_dummy_bundle() -> pd_data.PDPairBundle:
    pairs: List[PromptPair] = []
    for idx in range(4):
        meta = PairMeta(
            opt_a=f"opt_a_{idx}",
            opt_b=f"opt_b_{idx}",
            defect_label="A",
            cooperate_label="B",
            description=f"desc_{idx}",
        )
        pairs.append(
            PromptPair(
                positive=f"pos_{idx}",
                negative=f"neg_{idx}",
                meta=meta,
            )
        )
    # Simple split: first half train, second half test
    train_pairs = pairs[:2]
    test_pairs = pairs[2:]
    return pd_data.PDPairBundle(pairs=pairs, train_pairs=train_pairs, test_pairs=test_pairs)


def test_run_smoke(tmp_path: Path, monkeypatch):
    # Patch build_pd_pair_bundle to avoid reading real JSON
    monkeypatch.setattr(mod, "build_pd_pair_bundle", lambda *_args, **_kw: _make_dummy_bundle())

    # Patch AutoModelForCausalLM / AutoTokenizer to avoid HF loads
    monkeypatch.setattr(
        mod,
        "AutoModelForCausalLM",
        type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: _DummyModel(num_layers=2))}),
    )
    monkeypatch.setattr(
        mod,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: _DummyTokenizer())}),
    )

    # Patch train_pd_repreader to avoid RepReadingPipeline
    def _fake_train_pd_repreader(
        model,
        tokenizer,
        train_data: Dict[str, object],
        test_data: Dict[str, object],
        hidden_layers: Sequence[int],
        batch_size: int,
        max_length: int,
        span_mode: str = "assistant",
    ):
        # Return synthetic accuracies and vectors for 2 layers
        layer_acc = {layer: 0.8 + 0.1 * (1 if layer == 0 else 0) for layer in hidden_layers}
        layer_vectors = {layer: np.ones(4, dtype=np.float32) * (layer + 1) for layer in hidden_layers}
        rep_reader = object()
        return rep_reader, layer_acc, layer_vectors

    monkeypatch.setattr(mod, "train_pd_repreader", _fake_train_pd_repreader)

    # Patch _decision_rate and _register_control_hook to avoid heavy computation
    monkeypatch.setattr(mod, "_decision_rate", lambda *a, **k: 0.5)

    def _fake_register(layer, vec, intensity):
        class H:
            def remove(self):
                pass

        return H()

    monkeypatch.setattr(mod, "_register_control_hook", _fake_register)

    # Also patch _token_id to avoid tokenizer internals
    monkeypatch.setattr(mod, "_token_id", lambda _tok, _s: 0)

    out_dir = tmp_path / "pd_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = mod.run(
        model_path="dummy-model",
        output_dir=out_dir,
        max_length=64,
        batch_size=2,
        seed=0,
        intensity=1.0,
        max_pairs=2,
        middle_third_only=False,
        behavior_intensities=[0.5, 1.0],
    )

    # Basic structural checks on result
    assert "best_layer" in result
    assert "best_accuracy" in result
    assert "layer_accuracies" in result
    assert isinstance(result["best_layer"], int)
    assert isinstance(result["best_accuracy"], float)
    assert isinstance(result["layer_accuracies"], dict)

    # Our fake train_pd_repreader sets layer 0 to highest accuracy
    assert result["best_layer"] == 0
    assert result["layer_accuracies"][0] >= result["layer_accuracies"][1]

    # After unifying output layout, artifacts live under:
    # out_dir/<model>/<timestamp>/seed_<seed>/
    model_root = out_dir / "dummy-model"
    assert model_root.is_dir()
    ts_dirs = [p for p in model_root.iterdir() if p.is_dir()]
    assert ts_dirs, "Expected at least one timestamp directory under out_dir/<model>/"
    seed_dir = ts_dirs[0] / "seed_0"
    assert seed_dir.is_dir()
    assert (seed_dir / "result.json").exists()
    assert (seed_dir / "best_vector.npy").exists()

    # result.json should be parseable and consistent with returned result
    on_disk = json.loads((seed_dir / "result.json").read_text())
    assert on_disk["best_layer"] == result["best_layer"]
    assert abs(on_disk["best_accuracy"] - result["best_accuracy"]) < 1e-6

    # Layer vectors should be saved under the run directory
    vectors_dir = seed_dir / "layer_vectors"
    assert vectors_dir.is_dir()
    # Two layers in hidden_layers for the smoke test
    for layer_idx in (0, 1):
        path = vectors_dir / f"layer_{layer_idx}.npy"
        assert path.exists(), f"Missing vector file: {path}"
