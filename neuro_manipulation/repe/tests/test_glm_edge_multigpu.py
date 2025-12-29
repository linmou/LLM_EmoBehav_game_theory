import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.mark.gpu
def test_glm_edge_multigpu_handles_image_forward():
    """
    Smoke-test that GLM-Edge-V-2b can accept image+text inputs when sharded
    across multiple GPUs via device_map="auto".

    Skips when:
      - Fewer than 2 CUDA devices
      - Local model path not present
      - No image-capable processor available
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires >= 2 CUDA devices for multi-GPU sharding")

    # Default local path; override via env if needed
    model_path = os.environ.get(
        "GLM_EDGE_MODEL_PATH",
        "/data/home/jjl7137/huggingface_models/zai-org/glm-edge-v-2b",
    )
    if not Path(model_path).exists():
        pytest.skip(f"Model path not found: {model_path}")

    # Lazy import to avoid heavy deps when skipped
    from transformers import AutoProcessor, AutoModelForCausalLM

    # Load processor and ensure it is image-capable
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    has_image = hasattr(processor, "image_processor") or hasattr(
        processor, "feature_extractor"
    )
    if not has_image:
        pytest.skip("Processor lacks image capability; skipping multimodal shard test")

    # Load model with multi-GPU sharding
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    # Build a tiny test image + simple prompt
    img = Image.new("RGB", (64, 64), "blue")
    prompt = "Describe the image briefly."

    # Prepare inputs (CPU tensors; Accelerate will handle device movement)
    inputs = processor(text=[prompt], images=[img], return_tensors="pt")

    # Inference without generation to keep it light
    with torch.no_grad():
        try:
            _ = model(**inputs, output_hidden_states=False)
        except RuntimeError as e:
            # Surface common cross-device errors explicitly
            if "Expected all tensors to be on the same device" in str(e):
                pytest.fail(
                    "GLM-Edge failed multimodal forward under multi-GPU sharding: "
                    f"{e}"
                )
            raise

    # If we made it here, the model handled a simple multimodal forward on multi-GPU
    assert True


@pytest.mark.gpu
def test_glm_edge_single_gpu_fallback_multimodal():
    """
    Sanity check: forcing single-GPU placement should always work for multimodal.
    Demonstrates a safe fallback when multi-GPU forward is problematic.
    """
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    model_path = os.environ.get(
        "GLM_EDGE_MODEL_PATH",
        "/data/home/jjl7137/huggingface_models/zai-org/glm-edge-v-2b",
    )
    if not Path(model_path).exists():
        pytest.skip(f"Model path not found: {model_path}")

    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    has_image = hasattr(processor, "image_processor") or hasattr(
        processor, "feature_extractor"
    )
    if not has_image:
        pytest.skip("Processor lacks image capability; skipping multimodal test")

    # Force single-GPU placement
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    ).eval()

    img = Image.new("RGB", (64, 64), "green")
    prompt = "Describe the image briefly."
    inputs = processor(text=[prompt], images=[img], return_tensors="pt")

    with torch.no_grad():
        _ = model(**inputs, output_hidden_states=False)

    assert True

