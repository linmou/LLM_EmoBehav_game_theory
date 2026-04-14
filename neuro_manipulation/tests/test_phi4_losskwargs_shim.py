#!/usr/bin/env python
# Responsible file: neuro_manipulation/utils.py
# Purpose: verify Phi-4-mini-instruct loading injects a LossKwargs compatibility shim when the installed transformers package does not provide it.

from types import SimpleNamespace
from unittest.mock import patch

def test_load_model_only_injects_phi4_losskwargs_shim(monkeypatch):
    """I am starting with a failing test. This is the Red phase."""
    import transformers.utils as tf_utils

    from neuro_manipulation.utils import load_model_only

    class DummyModel:
        def eval(self):
            return self

    if hasattr(tf_utils, "LossKwargs"):
        monkeypatch.delattr(tf_utils, "LossKwargs", raising=False)

    def fake_from_pretrained(model_name_or_path, **kwargs):
        shim = getattr(tf_utils, "LossKwargs", None)
        assert shim is not None, "LossKwargs shim was not installed"
        assert issubclass(shim, dict), "LossKwargs shim must behave like a TypedDict"
        assert getattr(shim, "__annotations__", {}) == {}
        return DummyModel()

    with patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(
            architectures=["Phi3ForCausalLM"],
            model_type="phi3",
        ),
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=fake_from_pretrained,
    ), patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=AssertionError("AutoModel fallback should not be used"),
    ):
        model = load_model_only(
            model_name_or_path="/home/jjl7137/huggingface_models/microsoft/Phi-4-mini-instruct",
            from_vllm=False,
            loading_config={"dtype": "bfloat16"},
        )

    assert isinstance(model, DummyModel)
