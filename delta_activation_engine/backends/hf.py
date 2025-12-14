"""
Responsible: delta_activation_engine/backends/hf.py
Purpose: HF backend for computing last-layer, last-token representations.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseBackend


def select_middle_third_layers(total_layers: int) -> List[int]:
    """Return 0-based middle-third indices, e.g., 12 -> [4..7], 3 -> [1]."""
    if total_layers <= 0:
        return []
    start = total_layers // 3
    end = (2 * total_layers) // 3
    return list(range(start, end))


class HFBackend(BaseBackend):
    def __init__(self, cfg):
        # Deferred imports to limit GPU dependencies at import time
        from neuro_manipulation.utils import setup_model_and_tokenizer, load_tokenizer_only
        from neuro_manipulation.configs.experiment_config import get_repe_eng_config
        from neuro_manipulation.repe.wrapped_model import WrappedReadingVecModel
        from neuro_manipulation.model_layer_detector import ModelLayerDetector
        from neuro_manipulation.model_utils import load_emotion_readers

        self.cfg = cfg
        self.model, self.tokenizer, self.prompt_format, _ = setup_model_and_tokenizer(
            cfg.loading_config, from_vllm=False
        )
        self.model.eval()

        # Ensure hidden states are available
        if getattr(self.model.config, "output_hidden_states", None) is not True:
            try:
                self.model.config.output_hidden_states = True
            except Exception:
                pass

        num_layers = ModelLayerDetector.num_layers(self.model)
        self.control_layers = select_middle_third_layers(num_layers)

        # RepE config and readers
        self.repe_cfg = get_repe_eng_config(cfg.model_path, yaml_config=cfg.repe_eng_config)
        self.readers = load_emotion_readers(
            self.repe_cfg, self.model, self.tokenizer, self.control_layers, None, False
        )

        # Wrap model for reading-vec control
        self.wrapped = WrappedReadingVecModel(self.model, self.tokenizer)

        # Representation config
        self.max_length = 256

    def _forward_last_hidden_avg(self, texts: List[str]) -> np.ndarray:
        import torch
        import numpy as np
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        hs = out.hidden_states[-1]  # last hidden
        vecs = hs[:, -1, :]  # last token
        return vecs.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    def _forward_last_hidden_avg_steered(self, texts: List[str], emotion: str, intensity: float) -> np.ndarray:
        import numpy as np
        # Build direction as operator='linear_comb' across control layers
        readers = self.readers.get(emotion)
        if readers is None:
            raise ValueError(f"No rep readers for emotion '{emotion}'")

        # Minimal linear-comb: layer-wise vectors scaled by intensity
        steered = self.wrapped.forward_with_control(
            texts=texts,
            readers=readers,
            operator="linear_comb",
            intensity=float(intensity),
        )
        return steered

    def get_repr(
        self,
        prompts: List[str],
        *,
        steered: bool,
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        if not steered:
            return self._forward_last_hidden_avg(prompts)
        assert emotion is not None and intensity is not None
        return self._forward_last_hidden_avg_steered(prompts, emotion, float(intensity))

    def get_run_metadata(self) -> dict:
        return {
            "backend": "hf",
            "control_layers": self.control_layers,
            "max_length": self.max_length,
        }
