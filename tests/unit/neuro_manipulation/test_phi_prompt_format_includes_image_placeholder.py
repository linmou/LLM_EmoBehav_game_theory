# Responsible file: `tests/unit/neuro_manipulation/test_phi_prompt_format_includes_image_placeholder.py`
# Purpose: Phi VLM prompt formatting must include `<|image_1|>` so vLLM multimodal prompt replacement can succeed.

from __future__ import annotations

from neuro_manipulation.prompt_formats import ManualPromptFormat


def test_phi_vision_prompt_includes_image_placeholder() -> None:
    fmt = ManualPromptFormat.get("microsoft/Phi-3.5-vision-instruct")
    prompt = fmt.build(
        system_prompt="sys",
        user_messages=["hello"],
        assistant_messages=[],
        images=["img"],
    )
    assert "<|image_1|>" in prompt
    assert "<|assistant|>" in prompt
