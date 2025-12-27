"""
Responsible: delta_activation_engine/backends/hf.py
Purpose: Ensure HFBackend imports from neuro_manipulation.model_utils path and can be constructed with stubs.
"""

import importlib
import sys
import types


def test_hf_backend_imports_from_model_utils(monkeypatch):
    import numpy as np

    class DummyModel:
        def __init__(self):
            self.config = types.SimpleNamespace(output_hidden_states=True)
            self.device = "cpu"

        def eval(self):
            return None

    class DummyTokenizer:
        pass

    def setup_model_and_tokenizer(cfg, from_vllm=False):
        return DummyModel(), DummyTokenizer(), "pf", None

    def load_emotion_readers(repe_cfg, model, tokenizer, control_layers, *_args):
        return {"anger": "reader"}

    def forward_with_control(texts, readers, operator, intensity):
        return np.ones(4, dtype=np.float32) * float(intensity)

    stub_model_utils = types.ModuleType("neuro_manipulation.model_utils")
    stub_model_utils.setup_model_and_tokenizer = setup_model_and_tokenizer
    stub_model_utils.load_emotion_readers = load_emotion_readers

    stub_experiment_config = types.ModuleType("neuro_manipulation.configs.experiment_config")
    stub_experiment_config.get_repe_eng_config = lambda *a, **k: {}

    stub_wrapped = types.ModuleType("neuro_manipulation.repe.rep_control_reading_vec")

    class DummyWrappedReadingVecModel:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def forward_with_control(self, texts, readers, operator, intensity):
            return forward_with_control(texts, readers, operator, intensity)

    stub_wrapped.WrappedReadingVecModel = DummyWrappedReadingVecModel

    stub_layer_detector = types.ModuleType("neuro_manipulation.model_layer_detector")

    class DummyLayerDetector:
        @staticmethod
        def num_layers(model):
            return 12

    stub_layer_detector.ModelLayerDetector = DummyLayerDetector

    stub_pipelines = types.ModuleType("neuro_manipulation.repe.pipelines")
    stub_pipelines.repe_pipeline_registry = lambda: None

    monkeypatch.setitem(sys.modules, "neuro_manipulation.model_utils", stub_model_utils)
    monkeypatch.setitem(
        sys.modules, "neuro_manipulation.configs.experiment_config", stub_experiment_config
    )
    monkeypatch.setitem(sys.modules, "neuro_manipulation.repe.rep_control_reading_vec", stub_wrapped)
    monkeypatch.setitem(sys.modules, "neuro_manipulation.model_layer_detector", stub_layer_detector)
    monkeypatch.setitem(sys.modules, "neuro_manipulation.repe.pipelines", stub_pipelines)

    sys.modules.pop("delta_activation_engine.backends.hf", None)
    import delta_activation_engine.backends.hf as hf_mod

    hf_mod = importlib.reload(hf_mod)
    HFBackend = hf_mod.HFBackend

    cfg = types.SimpleNamespace(
        model_path="dummy",
        emotions=["anger"],
        intensities=[0.0, 1.0],
        output_dir="/tmp",
        loading_config={"model_path": "dummy"},
        repe_eng_config={},
    )

    backend = HFBackend(cfg)
    assert backend.control_layers  # should be derived from DummyLayerDetector
