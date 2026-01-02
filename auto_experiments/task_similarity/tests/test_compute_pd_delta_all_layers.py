"""
Responsible: auto_experiments/task_similarity/compute_pd_delta.py
Purpose: Ensure delta activation measurement covers all layers (final-token
         hidden state per layer) and saves corresponding NPZ keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .. import compute_pd_delta as mod


class _DummyTokenizer:
    def __call__(
        self,
        texts: Sequence[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 256,
        add_special_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del return_tensors, padding, truncation, max_length, add_special_tokens
        batch = len(texts)
        # Two tokens per prompt, no padding.
        input_ids = torch.ones((batch, 2), dtype=torch.long)
        attention_mask = torch.ones((batch, 2), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


_STEERED: Dict[str, bool] = {"on": False}


class _DummyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))


class _DummyModel(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden: int = 4) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = type("Cfg", (), {"num_hidden_layers": n_layers})

        class _Inner(torch.nn.Module):
            def __init__(self, layers: int) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([_DummyLayer() for _ in range(layers)])

        self.model = _Inner(n_layers)
        self.hidden = hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, output_hidden_states: bool = False, **_: Any):
        del attention_mask
        batch, seq = input_ids.shape
        assert output_hidden_states is True

        # hidden_states[0] is embedding; then layers 0..n-1.
        hs: List[torch.Tensor] = []
        hs.append(torch.zeros((batch, seq, self.hidden), dtype=torch.float32))
        for layer_idx in range(self.config.num_hidden_layers):
            base = float(layer_idx)
            steer = 1.0 if _STEERED["on"] else 0.0
            hs.append(torch.full((batch, seq, self.hidden), base + steer, dtype=torch.float32))
        return type("Out", (), {"hidden_states": tuple(hs)})


def test_compute_pd_delta_saves_all_layers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "AutoTokenizer", type("TokFactory", (), {"from_pretrained": staticmethod(lambda *a, **k: _DummyTokenizer())}))
    monkeypatch.setattr(mod, "AutoModelForCausalLM", type("ModelFactory", (), {"from_pretrained": staticmethod(lambda *a, **k: _DummyModel(n_layers=3))}))
    monkeypatch.setattr(mod, "get_generic_probes", lambda: ["p0", "p1"])

    def _fake_hook(_layer_module: Any, _vec: np.ndarray, _intensity: float):
        _STEERED["on"] = True

        class _Handle:
            @staticmethod
            def remove() -> None:
                _STEERED["on"] = False

        return _Handle()

    monkeypatch.setattr(mod, "_register_control_hook", _fake_hook)

    out_dir = tmp_path / "delta_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Provide a vector for control layer 0; content irrelevant because we patch hook.
    vec_dir = tmp_path / "vecs"
    vec_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_dir / "layer_0.npy", np.ones(4, dtype=np.float32))

    result = mod.run_delta(
        model_path="dummy-model",
        vector_path=vec_dir,
        layer=0,
        use_middle_third=False,
        intensity=1.0,
        output_dir=out_dir,
        max_length=16,
        batch_size=2,
        seed=0,
    )
    assert "measurement_layers" in result
    assert result["measurement_layers"] == [0, 1, 2]

    # One timestamped run dir should exist with delta.npz keys {0,1,2}
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert run_dirs
    delta_path = run_dirs[0] / "delta.npz"
    assert delta_path.exists()
    delta = np.load(delta_path)
    assert set(delta.files) == {"0", "1", "2"}
    # Baseline was layer_idx, steered adds +1.0 => delta should be all ones.
    for k in ("0", "1", "2"):
        assert np.allclose(delta[k], np.ones((4,), dtype=np.float32))
