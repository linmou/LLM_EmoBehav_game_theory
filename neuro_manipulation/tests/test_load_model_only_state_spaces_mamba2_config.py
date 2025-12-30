# Tests for `neuro_manipulation/utils.py::load_model_only` state-spaces Mamba2 config translation.
#
# Responsible file: neuro_manipulation/utils.py
# Purpose: state-spaces Mamba2 checkpoints ship a non-Transformers `config.json` (e.g. `d_model`, `n_layer`).
#          Our loader must translate it to `transformers.Mamba2Config(hidden_size=..., num_hidden_layers=...)`
#          so weights load without shape mismatches.

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestLoadModelOnlyStateSpacesMamba2Config(unittest.TestCase):
    def test_translates_state_spaces_mamba2_config_to_transformers_config(self):
        from neuro_manipulation.utils import load_model_only

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            # minimal checkpoint to allow inferring mamba2 num_heads/head_dim/vocab_size
            import torch

            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "d_model": 1536,
                        "n_layer": 48,
                        "vocab_size": 50277,
                        "ssm_cfg": {"layer": "Mamba2"},
                        "rms_norm": True,
                        "residual_in_fp32": True,
                        "tie_embeddings": True,
                    }
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    "backbone.embedding.weight": torch.zeros((50288, 1536)),
                    "backbone.layers.0.mixer.dt_bias": torch.zeros((48,)),
                    "backbone.layers.0.mixer.conv1d.weight": torch.zeros((3328, 1, 4)),
                    "backbone.layers.0.mixer.in_proj.weight": torch.zeros((6448, 1536)),
                },
                model_dir / "pytorch_model.bin",
            )

            captured = {}

            class _FakeMamba2Config:
                model_type = "mamba2"

                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            fake_transformers = types.SimpleNamespace()
            fake_transformers.AutoConfig = MagicMock()
            fake_transformers.AutoModel = MagicMock()
            fake_transformers.Mamba2Config = _FakeMamba2Config
            fake_transformers.MambaConfig = MagicMock()

            def _fake_from_pretrained(*args, **kwargs):
                captured.update(kwargs)
                return MagicMock(eval=MagicMock(return_value="MODEL"))

            fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=_fake_from_pretrained)
            fake_transformers.Mamba2ForCausalLM = types.SimpleNamespace(from_pretrained=_fake_from_pretrained)
            fake_transformers.MambaForCausalLM = types.SimpleNamespace(from_pretrained=_fake_from_pretrained)

            with patch.dict("sys.modules", {"transformers": fake_transformers}):
                out = load_model_only(model_name_or_path=str(model_dir), from_vllm=False)

            self.assertEqual(out, "MODEL")
            self.assertIn("config", captured)
            cfg = captured["config"]
            self.assertEqual(getattr(cfg, "hidden_size", None), 1536)
            self.assertEqual(getattr(cfg, "num_hidden_layers", None), 48)
            # Transformers Mamba2 needs num_heads/head_dim consistent with hidden_size*expand.
            self.assertEqual(getattr(cfg, "num_heads", None), 48)
            self.assertEqual(getattr(cfg, "head_dim", None), 64)
            self.assertEqual(getattr(cfg, "expand", None), 2)
            self.assertEqual(getattr(cfg, "n_groups", None), 1)
            self.assertEqual(getattr(cfg, "state_size", None), 128)
            # state-spaces checkpoints often pad vocab to a multiple (e.g. 16)
            self.assertEqual(getattr(cfg, "vocab_size", None), 50288)
            # state-spaces uses `backbone.embedding.weight` while Transformers expects `backbone.embeddings.weight`.
            self.assertIn("state_dict", captured)
            sd = captured["state_dict"]
            self.assertIn("backbone.embeddings.weight", sd)
            self.assertNotIn("backbone.embedding.weight", sd)
