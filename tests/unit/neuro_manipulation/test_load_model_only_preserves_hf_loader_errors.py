# Responsible file: `neuro_manipulation/utils.py`
# Purpose: HF multimodal loader failures must preserve the original dependency/config
# error instead of falling through to `AutoModel`, which obscures the real bug.

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_load_model_only_does_not_fallback_to_automodel_after_causallm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I am starting with a failing test. This is the Red phase."""
    import neuro_manipulation.utils as nm_utils
    import transformers as tf

    config = SimpleNamespace(architectures=["Phi4MMForCausalLM"])

    def fake_config_from_pretrained(*args, **kwargs):
        return config

    def fake_causallm_from_pretrained(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'backoff'")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("AutoModel fallback should not be used after HF loader failure")

    monkeypatch.setattr(tf, "AutoConfig", SimpleNamespace(from_pretrained=fake_config_from_pretrained))
    monkeypatch.setattr(tf, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=fake_causallm_from_pretrained))
    monkeypatch.setattr(tf, "AutoModel", SimpleNamespace(from_pretrained=fail_if_called))
    with pytest.raises(ModuleNotFoundError, match="backoff"):
        nm_utils.load_model_only("microsoft/Phi-4-multimodal-instruct", from_vllm=False)
