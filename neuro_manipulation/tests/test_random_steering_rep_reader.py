"""
Responsible files:
- neuro_manipulation/repe/rep_readers.py
- neuro_manipulation/utils.py
- neuro_manipulation/model_utils.py

Purpose:
- ensure random steering vectors come from configurable normal distribution;
- ensure direction_finder_kwargs flows into RepReader construction;
- ensure random reader seeds are deterministic but varied across emotions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np

# Lightweight dependency stubs for local test environments without full ML stack.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace()
if "huggingface_hub" not in sys.modules:
    sys.modules["huggingface_hub"] = types.SimpleNamespace(hf_hub_download=lambda *args, **kwargs: "")
if "matplotlib" not in sys.modules:
    sys.modules["matplotlib"] = types.ModuleType("matplotlib")
if "matplotlib.pyplot" not in sys.modules:
    sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
if "PIL" not in sys.modules:
    pil_mod = types.ModuleType("PIL")
    pil_image_mod = types.ModuleType("PIL.Image")
    pil_image_mod.Image = object
    pil_mod.Image = pil_image_mod
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = pil_image_mod
if "tqdm" not in sys.modules:
    sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda x: x)
if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.SimpleNamespace(
        AutoModel=object,
        AutoTokenizer=object,
        MistralForCausalLM=object,
        pipeline=lambda *args, **kwargs: None,
    )

_REP_READERS_PATH = Path(__file__).resolve().parents[1] / "repe" / "rep_readers.py"
_SPEC = importlib.util.spec_from_file_location("nm_rep_readers", _REP_READERS_PATH)
assert _SPEC and _SPEC.loader
_REP_READERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REP_READERS)
RandomRepReader = _REP_READERS.RandomRepReader

_UTILS_PATH = Path(__file__).resolve().parents[1] / "utils.py"
_UTILS_SPEC = importlib.util.spec_from_file_location("nm_utils", _UTILS_PATH)
assert _UTILS_SPEC and _UTILS_SPEC.loader
_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(_UTILS)
all_emotion_rep_reader = _UTILS.all_emotion_rep_reader
get_rep_reader = _UTILS.get_rep_reader


def test_random_rep_reader_normal_params_and_seed() -> None:
    model = MagicMock()
    model.config.hidden_size = 2048

    reader_a = RandomRepReader(
        needs_hiddens=False, mean=0.4, std=0.3, seed=11, normalize_l2=False
    )
    reader_b = RandomRepReader(
        needs_hiddens=False, mean=0.4, std=0.3, seed=11, normalize_l2=False
    )
    dir_a = reader_a.get_rep_directions(model, None, None, hidden_layers=[-1])[-1][0]
    dir_b = reader_b.get_rep_directions(model, None, None, hidden_layers=[-1])[-1][0]

    assert np.allclose(dir_a, dir_b)
    assert abs(float(np.mean(dir_a)) - 0.4) < 0.05
    assert abs(float(np.std(dir_a)) - 0.3) < 0.05


def test_random_rep_reader_defaults_to_unit_norm_direction() -> None:
    model = MagicMock()
    model.config.hidden_size = 2048

    reader = RandomRepReader(needs_hiddens=False, mean=0.0, std=1.0, seed=7)
    direction = reader.get_rep_directions(model, None, None, hidden_layers=[-1])[-1][0]

    assert np.isfinite(direction).all()
    assert abs(float(np.linalg.norm(direction)) - 1.0) < 1e-4


def test_get_rep_reader_passes_direction_finder_kwargs() -> None:
    mock_pipeline = MagicMock()
    mock_reader = MagicMock()
    mock_reader.direction_signs = {-1: 1}
    mock_reader.directions = {-1: np.zeros((1, 8), dtype=np.float32)}
    mock_pipeline.get_directions.return_value = mock_reader

    with patch("neuro_manipulation.utils.test_direction", return_value=({-1: 1.0}, {})):
        get_rep_reader(
            rep_reading_pipeline=mock_pipeline,
            train_data={"data": ["a", "b"], "labels": [[1, 0], [0, 1]]},
            test_data={"data": ["c"], "labels": [[1, 0]]},
            hidden_layers=[-1],
            rep_token=-1,
            n_difference=1,
            direction_method="random",
            direction_finder_kwargs={"needs_hiddens": False, "mean": 0.0, "std": 1.0, "seed": 5},
        )

    _, kwargs = mock_pipeline.get_directions.call_args
    assert kwargs["direction_finder_kwargs"] == {
        "needs_hiddens": False,
        "mean": 0.0,
        "std": 1.0,
        "seed": 5,
    }


def test_all_emotion_rep_reader_offsets_seed_per_emotion() -> None:
    captured_kwargs: list[dict] = []

    def _fake_get_rep_reader(**kwargs):
        captured_kwargs.append(dict(kwargs["direction_finder_kwargs"]))
        fake_reader = MagicMock()
        return fake_reader, {-1: 1.0}

    data = {
        "anger": {"train": {"data": [], "labels": []}, "test": {"data": [], "labels": []}},
        "sadness": {"train": {"data": [], "labels": []}, "test": {"data": [], "labels": []}},
        "fear": {"train": {"data": [], "labels": []}, "test": {"data": [], "labels": []}},
    }

    with patch.object(_UTILS, "get_rep_reader", side_effect=_fake_get_rep_reader):
        all_emotion_rep_reader(
            data=data,
            emotions=["anger", "sadness", "fear"],
            rep_reading_pipeline=MagicMock(),
            hidden_layers=[-1],
            rep_token=-1,
            n_difference=1,
            direction_method="random",
            direction_finder_kwargs={"needs_hiddens": False, "mean": 0.0, "std": 1.0, "seed": 42},
            save_path=None,
            read_args=None,
        )

    assert captured_kwargs == [
        {"needs_hiddens": False, "mean": 0.0, "std": 1.0, "seed": 42},
        {"needs_hiddens": False, "mean": 0.0, "std": 1.0, "seed": 43},
        {"needs_hiddens": False, "mean": 0.0, "std": 1.0, "seed": 44},
    ]
