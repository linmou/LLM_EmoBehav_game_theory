# Responsible file: `neuro_manipulation/utils.py`
# Purpose: Gemma 3 multimodal checkpoints must use the HF image-text loader path that matches current transformers mappings.

from __future__ import annotations

from types import SimpleNamespace


def test_load_model_only_routes_gemma3_conditional_generation_to_image_text_loader(
    monkeypatch,
) -> None:
    import neuro_manipulation.utils as nm_utils
    import transformers as tf

    captured = {}

    class DummyModel:
        def eval(self):
            return self

    def fake_config_from_pretrained(*args, **kwargs):
        return SimpleNamespace(
            architectures=["Gemma3ForConditionalGeneration"],
            model_type="gemma3",
        )

    def fake_image_text_loader(*args, **kwargs):
        captured["loader"] = "image_text"
        captured["kwargs"] = kwargs
        return DummyModel()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Gemma 3 should not be loaded through Vision2Seq")

    monkeypatch.setattr(tf, "AutoConfig", SimpleNamespace(from_pretrained=fake_config_from_pretrained))
    monkeypatch.setattr(tf, "AutoModelForImageTextToText", SimpleNamespace(from_pretrained=fake_image_text_loader))
    monkeypatch.setattr(tf, "AutoModelForVision2Seq", SimpleNamespace(from_pretrained=fail_if_called))
    monkeypatch.setattr(tf, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=fail_if_called))
    monkeypatch.setattr(tf, "AutoModel", SimpleNamespace(from_pretrained=fail_if_called))

    model = nm_utils.load_model_only("google/gemma-3-4b-it", from_vllm=False)

    assert model is not None
    assert captured["loader"] == "image_text"
