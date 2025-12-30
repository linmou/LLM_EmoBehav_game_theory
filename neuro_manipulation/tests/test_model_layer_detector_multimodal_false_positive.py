"""
Tests for `neuro_manipulation/model_layer_detector.py`.

Purpose: Avoid false-positive multimodal detection on text-only models that have
embedding layers (common attribute names include 'embed_tokens'). A false
positive can cause layer detection to pick adapter/fusion ModuleLists instead of
the actual transformer/mamba layer stack.
"""

import unittest

import torch.nn as nn


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Linear(4, 4)


class _FakeTextOnlyModelWithAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        # A normal text model will have embeddings; this should NOT trigger multimodal.
        self.embed_tokens = nn.Embedding(10, 4)
        # The real stack of layers we want.
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(3)])
        # An adapter list that should not be preferred.
        self.adapter_list = nn.ModuleList([nn.Linear(4, 8) for _ in range(3)])


class TestModelLayerDetectorMultimodalFalsePositive(unittest.TestCase):
    def test_is_multimodal_model_does_not_trigger_on_embed_tokens(self):
        from neuro_manipulation.model_layer_detector import ModelLayerDetector

        m = _FakeTextOnlyModelWithAdapters()
        self.assertFalse(ModelLayerDetector.is_multimodal_model(m))

    def test_get_model_layers_prefers_main_layers_over_adapter_list(self):
        from neuro_manipulation.model_layer_detector import ModelLayerDetector

        m = _FakeTextOnlyModelWithAdapters()
        layers = ModelLayerDetector.get_model_layers(m)
        self.assertIs(layers, m.layers)


class _FakeMambaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = nn.Linear(4, 4)


class Qwen2VLForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(3)])
        self.adapter_list = nn.ModuleList([nn.Linear(4, 8) for _ in range(3)])


class _FakeMambaOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeMambaLayer() for _ in range(3)])


class TestModelLayerDetectorArchitectureAgnostic(unittest.TestCase):
    def test_get_model_layers_prefers_layers_even_when_model_is_multimodal(self):
        from neuro_manipulation.model_layer_detector import ModelLayerDetector

        m = Qwen2VLForConditionalGeneration()
        self.assertTrue(ModelLayerDetector.is_multimodal_model(m))
        self.assertIs(ModelLayerDetector.get_model_layers(m), m.layers)

    def test_get_model_layers_detects_mamba_style_blocks(self):
        from neuro_manipulation.model_layer_detector import ModelLayerDetector

        m = _FakeMambaOnlyModel()
        self.assertIs(ModelLayerDetector.get_model_layers(m), m.layers)

    def test_multimodal_prefers_language_layers_over_vision_encoder_layers(self):
        from neuro_manipulation.model_layer_detector import ModelLayerDetector

        class Qwen2VLForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = nn.Module()
                self.vision_model.encoder = nn.Module()
                self.vision_model.encoder.layers = nn.ModuleList(
                    [nn.Identity() for _ in range(2)]
                )
                self.language_model = nn.Module()
                self.language_model.model = nn.Module()
                self.language_model.model.layers = nn.ModuleList(
                    [_FakeLayer() for _ in range(3)]
                )

        m = Qwen2VLForConditionalGeneration()
        self.assertTrue(ModelLayerDetector.is_multimodal_model(m))
        self.assertIs(ModelLayerDetector.get_model_layers(m), m.language_model.model.layers)

    def test_multimodal_does_not_accidentally_select_vision_layers_that_look_transformerish(self):
        from neuro_manipulation.model_layer_detector import ModelLayerDetector

        class VisionLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Linear(4, 4)

        class Qwen2VLForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = nn.Module()
                self.vision_model.encoder = nn.Module()
                self.vision_model.encoder.layers = nn.ModuleList(
                    [VisionLayer() for _ in range(2)]
                )
                self.language_model = nn.Module()
                self.language_model.model = nn.Module()
                self.language_model.model.layers = nn.ModuleList(
                    [_FakeLayer() for _ in range(3)]
                )

        m = Qwen2VLForConditionalGeneration()
        self.assertTrue(ModelLayerDetector.is_multimodal_model(m))
        self.assertIs(ModelLayerDetector.get_model_layers(m), m.language_model.model.layers)
