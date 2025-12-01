"""
Smoke test for run_pd_defection_pd_behavior.run.

Responsible: auto_experiments/task_similarity/run_pd_defection_pd_behavior.py
Purpose: Ensure the PD → game_theory behavior runner wires together
         config loading, activation spec, dataset iteration, and
         intensity-dependent decision ratios without touching real
         HF models or benchmark components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .. import run_pd_defection_pd_behavior as mod


class _DummyTokenizer:
    def __init__(self) -> None:
        self.name_or_path = "dummy-tokenizer"

    def __call__(
        self,
        texts: Sequence[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 256,
        add_special_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del truncation, max_length, add_special_tokens  # unused in dummy
        batch_size = len(texts)
        # Single fake token per prompt
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(
        self,
        sequences: torch.Tensor | Sequence[Sequence[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        del sequences, skip_special_tokens  # decoding ignores token ids
        # Generate responses based on CURRENT_INTENSITY set by fake hook
        val = _CURRENT_INTENSITY["value"]
        if val > 0.0:
            decision = "defect"
        else:
            decision = "cooperate"
        return [f'{{"decision": "{decision}"}}'] * _CURRENT_BATCH_SIZE["value"]


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Two fake layers; parameters unused because we patch _register_control_hook
        self.config = type("Cfg", (), {"num_hidden_layers": 2})

        class Inner:
            def __init__(self) -> None:
                self.layers = [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]

        self.model = Inner()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **_: Any,
    ) -> torch.Tensor:
        # Record batch size so tokenizer.batch_decode can match it
        del attention_mask, max_new_tokens, do_sample, temperature, top_p
        _CURRENT_BATCH_SIZE["value"] = int(input_ids.size(0))
        # Return dummy token ids; tokenizer ignores content
        return torch.zeros((input_ids.size(0), 1), dtype=torch.long)


class _DummyDataset:
    # Reuse the real extraction helpers to keep parsing semantics realistic
    from emotion_experiment_engine.datasets.games import GameTheoryDataset as _RealGT

    _extract_options_from_prompt = staticmethod(_RealGT._extract_options_from_prompt)
    _extract_option_from_response = staticmethod(_RealGT._extract_option_from_response)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        # Three simple prompts with PD-style option lines
        self.items = [
            type(
                "Item",
                (),
                {
                    "id": f"pd-{i}",
                    "input_text": "Prisoners dilemma event",
                    "context": None,
                    "ground_truth": None,
                    "metadata": {
                        "options": [
                            {"id": 1, "text": "Cooperate"},
                            {"id": 2, "text": "Defect"},
                        ]
                    },
                },
            )()
            for i in range(3)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        # Minimal prompt containing option lines expected by extraction helpers
        prompt = (
            f"Scenario: {item.input_text}\n"
            "Option 1. Cooperate\n"
            "Option 2. Defect\n"
        )
        return {"item": item, "prompt": prompt, "ground_truth": None}


_CURRENT_INTENSITY: Dict[str, float] = {"value": 0.0}
_CURRENT_BATCH_SIZE: Dict[str, int] = {"value": 0}


def test_run_pd_behavior_smoke(tmp_path: Path, monkeypatch) -> None:
    """
    High-level smoke test for run():
    - Patches HF model/tokenizer and benchmark components.
    - Uses fake hook to track intensity.
    - Verifies that defect ratio increases with intensity.
    """

    # Patch tokenizer / model constructors
    monkeypatch.setattr(
        mod,
        "AutoTokenizer",
        type("TokFactory", (), {"from_pretrained": staticmethod(lambda *a, **k: _DummyTokenizer())}),
    )
    monkeypatch.setattr(
        mod,
        "AutoModelForCausalLM",
        type("ModFactory", (), {"from_pretrained": staticmethod(lambda *a, **k: _DummyModel())}),
    )

    # Patch PromptFormat (not used by dummy dataset, but keep API surface small)
    class _DummyPromptFormat:
        def __init__(self, tokenizer: Any) -> None:
            self.tokenizer = tokenizer

        def build(self, system_prompt: str, user_messages: List[str], assistant_messages: List[str] | None = None, images=None, enable_thinking: bool = False) -> str:  # type: ignore[override]
            del assistant_messages, images, enable_thinking
            # Simple concatenation; not used by dummy dataset
            return system_prompt + "\n" + "\n".join(user_messages)

    monkeypatch.setattr(mod, "PromptFormat", _DummyPromptFormat)

    # Patch GameTheoryDataset to use dummy dataset implementation
    monkeypatch.setattr(mod, "GameTheoryDataset", _DummyDataset)

    # Patch hook registration to track current intensity
    def _fake_register(layer: Any, vec: np.ndarray, intensity: float):
        del layer, vec
        _CURRENT_INTENSITY["value"] = float(intensity)

        class _Handle:
            @staticmethod
            def remove() -> None:
                _CURRENT_INTENSITY["value"] = 0.0

        return _Handle()

    monkeypatch.setattr(mod, "_register_control_hook", _fake_register)

    # Prepare minimal benchmark config YAML
    cfg_path = tmp_path / "pd_behavior_config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: game_theory",
                "task_type: Prisoners_Dilemma",
                "sample_limit: null",
                "augmentation_config: null",
                "enable_auto_truncation: false",
                "truncation_strategy: right",
                "preserve_ratio: 1.0",
                "llm_eval_config: null",
                "base_data_dir: null",
                "data_path: null",
                "generation_config:",
                "  max_new_tokens: 16",
                "  temperature: 0.0",
                "  top_p: 1.0",
                "  do_sample: false",
                "batch_size: 2",
            ]
        ),
        encoding="utf-8",
    )

    # Prepare activation spec and vector file
    pd_result_dir = tmp_path / "pd_run"
    pd_result_dir.mkdir(parents=True, exist_ok=True)
    vec_path = pd_result_dir / "best_vector.npy"
    np.save(vec_path, np.ones(4, dtype=np.float32))

    spec_path = tmp_path / "activation_spec.json"
    spec = {
        "pd_result_dir": str(pd_result_dir),
        "layer": 0,
        "vector_path": "best_vector.npy",
        "span_mode": "option",
        "pd_best_layer": 0,
        "pd_best_accuracy": 0.75,
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out_dir = tmp_path / "pd_behavior_out"

    result = mod.run(
        model_path="dummy-model",
        benchmark_config_path=cfg_path,
        activation_spec_path=spec_path,
        output_dir=out_dir,
        intensities=[0.0, 1.0],
        max_length=64,
        batch_size=2,
        seed=0,
    )

    # Structural checks
    assert result["model_path"] == "dummy-model"
    assert result["benchmark_name"] == "game_theory"
    assert result["task_type"] == "Prisoners_Dilemma"
    assert result["pd_best_layer"] == 0
    assert result["pd_best_accuracy"] == 0.75
    assert result["n_items"] == 3

    ratios = result["defect_ratio"]
    assert 0.0 in ratios and 1.0 in ratios

    # Our dummy pipeline: baseline (0.0) picks cooperate, steered (1.0) picks defect
    assert ratios[0.0] == 0.0
    assert ratios[1.0] == 1.0

    # Summary JSON should be written under output_dir
    summaries = list(out_dir.glob("*.json"))
    assert summaries, "Expected at least one behavior summary JSON file"
