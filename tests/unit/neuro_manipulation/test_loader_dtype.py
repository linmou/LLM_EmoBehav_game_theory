#!/usr/bin/env python3
# tests/unit/neuro_manipulation/test_loader_dtype.py - verifies HF loader honors dtype config

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


# Ensure project root on sys.path for direct imports
project_root = Path(__file__).parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Provide a lightweight vllm stub so module import succeeds in test env
if "vllm" not in sys.modules:
    vllm_stub = MagicMock()
    vllm_stub.LLM = MagicMock()
    sys.modules["vllm"] = vllm_stub

from neuro_manipulation.utils import load_model_only


def test_load_model_only_uses_bfloat16_when_configured():
    """load_model_only should respect dtype=bfloat16 for HF models."""

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    loading_config = SimpleNamespace(dtype="bfloat16")

    with patch("transformers.AutoConfig.from_pretrained") as mock_auto_config, \
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_causal_lm, \
        patch("transformers.AutoModel.from_pretrained") as mock_auto_model:

        mock_auto_config.return_value = SimpleNamespace(architectures=["StubForCausalLM"])
        mock_causal_lm.return_value = mock_model

        result = load_model_only("stub-model", from_vllm=False, loading_config=loading_config)

    assert result is mock_model
    assert mock_causal_lm.call_args.kwargs["torch_dtype"] == torch.bfloat16
    mock_auto_model.assert_not_called()
