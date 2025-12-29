import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
from PIL import Image

# Ensure repository root is on path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline
from neuro_manipulation.repe.vlm_adapters import AdapterContext


@pytest.fixture(autouse=True)
def cpu_only_repe_ops(monkeypatch):
    """Force rep_readers ops to use CPU to avoid CUDA in CI/test envs."""
    import neuro_manipulation.repe.rep_readers as rr

    def recenter_cpu(x, mean=None):
        xt = torch.tensor(x)
        if mean is None:
            mean_t = torch.mean(xt, axis=0, keepdims=True)
        else:
            mean_t = torch.tensor(mean)
        return xt - mean_t

    def project_cpu(H, direction):
        Ht = torch.tensor(H) if not isinstance(H, torch.Tensor) else H
        dt = torch.tensor(direction) if not isinstance(direction, torch.Tensor) else direction
        mag = torch.norm(dt)
        return Ht.matmul(dt) / mag

    monkeypatch.setattr(rr, "recenter", recenter_cpu, raising=True)
    monkeypatch.setattr(rr, "project_onto_direction", project_cpu, raising=True)

class FakeProcessor:
    def __init__(self) -> None:
        pass

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<formatted>"

    def __call__(self, text=None, images=None, videos=None, padding=True, return_tensors="pt", **kwargs):
        out = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        if images is not None:
            out["pixel_values"] = torch.ones(1, 3, 8, 8)
        return out


class FakeTokenizer:
    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path

    def __call__(self, text, return_tensors="pt", padding=True, **kwargs):
        return {"input_ids": torch.ones(1, 4, dtype=torch.long)}


class FakeModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 16, n_layers: int = 6) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # Minimal config object to satisfy transformers.Pipeline init
        self.config = type("Cfg", (), {})()

    def forward(self, **model_inputs):
        # Derive batch and seq_len from input_ids
        input_ids = model_inputs.get("input_ids")
        batch = int(input_ids.shape[0]) if isinstance(input_ids, torch.Tensor) else 1
        seq_len = int(input_ids.shape[1]) if isinstance(input_ids, torch.Tensor) else 4
        # Build list of layer tensors
        hiddens = [
            torch.randn(batch, seq_len, self.hidden_size) for _ in range(self.n_layers)
        ]
        return {"hidden_states": hiddens}


def build_pipeline(model_name: str) -> RepReadingPipeline:
    pipeline = RepReadingPipeline(model=FakeModel(), tokenizer=FakeTokenizer(model_name))
    # Attach an image processor/AutoProcessor equivalent
    pipeline.image_processor = FakeProcessor()
    return pipeline


@pytest.mark.parametrize(
    "model_name, supports_images",
    [
        ("Qwen/Qwen2.5-VL-3B-Instruct", True),
        ("openbmb/MiniCPM-V-4", True),
        ("zai-org/glm-edge-v-2b", True),
        ("google/gemma-3-4b-it", False),
        ("microsoft/Phi-3.5-mini-instruct", False),
    ],
)
def test_adapter_integration_preprocess_and_forward(model_name: str, supports_images: bool):
    pipeline = build_pipeline(model_name)

    image = Image.new("RGB", (16, 16), "red")
    inp = {"text": "when you see this image, your emotion is anger", "images": [image]}

    # Prepare via adapters in pipeline
    model_inputs = pipeline.preprocess(inp)

    assert isinstance(model_inputs, dict)
    assert "input_ids" in model_inputs
    if supports_images:
        assert "pixel_values" in model_inputs
    else:
        assert "pixel_values" not in model_inputs

    # Use adapter-based default rep_token if set to auto
    _, fwd, _ = pipeline._sanitize_parameters(
        rep_reader=None, rep_token="auto", hidden_layers=[-1, -2], which_hidden_states=None
    )
    out = pipeline._forward(model_inputs, fwd["rep_token"], fwd["hidden_layers"], rep_reader=None)

    assert -1 in out and -2 in out
    for k in [-1, -2]:
        t = out[k]
        assert isinstance(t, torch.Tensor)
        assert t.ndim == 2 and t.shape[0] == 1  # [batch, hidden_size]


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "openbmb/MiniCPM-V-4",
        "zai-org/glm-edge-v-2b",
        "google/gemma-3-4b-it",
        "microsoft/Phi-3.5-mini-instruct",
    ],
)
def test_end_to_end_get_directions_minimal(model_name: str):
    """Full pipeline path for get_directions on tiny synthetic data.

    This simulates the behavior used by the Qwen2.5 multimodal config by
    creating a handful of multimodal inputs and learning PCA directions.
    """
    pipeline = build_pipeline(model_name)

    # Create small synthetic dataset (4 samples → 2 pairs for n_difference=1)
    img = Image.new("RGB", (16, 16), "green")
    samples = [
        {"text": "when you see this image, your emotion is happiness", "images": [img]},
        {"text": "when you see this image, your emotion is sadness", "images": [img]},
        {"text": "when you see this image, your emotion is anger", "images": [img]},
        {"text": "when you see this image, your emotion is fear", "images": [img]},
    ]

    df = pipeline.get_directions(
        train_inputs=samples,
        rep_token="auto",
        hidden_layers=[-1, -2],
        n_difference=1,
        batch_size=2,
        train_labels=None,  # no sign computation for simplicity
        direction_method="pca",
    )

    assert hasattr(df, "directions")
    assert -1 in df.directions and -2 in df.directions
    for k in [-1, -2]:
        d = df.directions[k]
        # Expect shape (n_components, hidden_size) → (1, 16)
        assert getattr(d, "shape", None) is not None
        assert d.shape[0] == 1


def test_simulated_config_qwen2_5_mm_prisoners():
    """Simulate critical parts of config/qwen2.5_MM_Series_Prisoners_Dilemm.yaml.

    We don't run the full game pipeline; we validate that the multimodal
    repe_config-style inputs execute end-to-end on a small synthetic batch.
    """
    # Use a Qwen-VL model path to trigger multimodal adapter
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    pipeline = build_pipeline(model_name)

    # Simulated repe_config content
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    img = Image.new("RGB", (16, 16), "blue")

    # Build minimal stimuli (one sample per emotion)
    samples = [
        {"text": f"when you see this image, your emotion is {e}", "images": [img]}
        for e in emotions
    ]

    # Learn directions (PCA), no labels/signs
    df = pipeline.get_directions(
        train_inputs=samples,
        rep_token="auto",
        hidden_layers=[-1],
        n_difference=1,
        batch_size=3,
        train_labels=None,
        direction_method="pca",
    )

    assert -1 in df.directions
    d = df.directions[-1]
    assert d.shape[0] == 1
