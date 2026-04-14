# Responsible file: `neuro_manipulation/utils.py`
# Purpose: Phi vision/multimodal loaders should override `flash_attention_2`
# when `flash_attn` is unavailable, so loading fails only on real model/runtime
# constraints instead of a missing optional extension.

from __future__ import annotations

from types import SimpleNamespace


def test_phi_loader_forces_safe_attention_backend_without_flash_attn(monkeypatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    import neuro_manipulation.utils as nm_utils
    import transformers as tf

    captured: dict[str, object] = {}

    class DummyModel:
        def eval(self):
            return self

    def fake_config_from_pretrained(*args, **kwargs):
        return SimpleNamespace(architectures=["Phi3VForCausalLM"])

    def fake_causallm_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(tf, "AutoConfig", SimpleNamespace(from_pretrained=fake_config_from_pretrained))
    monkeypatch.setattr(tf, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=fake_causallm_from_pretrained))
    monkeypatch.setattr(tf, "AutoModel", SimpleNamespace(from_pretrained=fake_causallm_from_pretrained))
    monkeypatch.setattr(nm_utils.importlib.util, "find_spec", lambda name: None if name == "flash_attn" else object())

    model = nm_utils.load_model_only("microsoft/Phi-3.5-vision-instruct", from_vllm=False)

    assert isinstance(model, DummyModel)
    assert captured["attn_implementation"] == "eager"
