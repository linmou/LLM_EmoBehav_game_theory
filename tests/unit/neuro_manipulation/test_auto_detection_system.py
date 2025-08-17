#!/usr/bin/env python3
"""
Comprehensive Tests for New Auto-Detection System

Tests the complete auto-detection pipeline:
1. Model capability detection
2. Data type detection (text vs image paths)
3. Experiment feasibility validation
4. Automatic processor loading
5. End-to-end integration
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

# Add project root to path for imports
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from neuro_manipulation.utils import (
    auto_load_processor,
    detect_emotion_data_type,
    detect_multimodal_model,
    load_model_tokenizer,
    primary_emotions_concept_dataset,
    validate_multimodal_experiment_feasibility,
)


class TestMultimodalModelDetection:
    """Test suite for multimodal model detection."""

    def test_detect_vl_models_by_name(self):
        """Test detection of VL models by name patterns."""
        multimodal_models = [
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "meta-llama/LLaVA-1.5-7B",
            "microsoft/BLIP-2",
            "openai/clip-vit-base-patch32",
            "GPT-4V",
            "some/multimodal-model",
        ]

        for model in multimodal_models:
            assert detect_multimodal_model(
                model
            ), f"Should detect {model} as multimodal"

    def test_detect_text_only_models(self):
        """Test detection of text-only models."""
        text_models = [
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "microsoft/DialoGPT-medium",
            "gpt2",
            "bert-base-uncased",
            "some-regular-model",
        ]

        for model in text_models:
            assert not detect_multimodal_model(
                model
            ), f"Should detect {model} as text-only"

    @patch("transformers.AutoProcessor")
    def test_detect_by_processor_availability(self, mock_auto_processor):
        """Test detection via AutoProcessor availability."""
        # Mock processor with image processing capability
        mock_processor = Mock()
        mock_processor.image_processor = Mock()
        mock_auto_processor.from_pretrained.return_value = mock_processor

        result = detect_multimodal_model("unknown/model")
        assert result == True

        # Mock processor without image processing
        mock_processor_text = Mock()
        (
            delattr(mock_processor_text, "image_processor")
            if hasattr(mock_processor_text, "image_processor")
            else None
        )
        mock_auto_processor.from_pretrained.return_value = mock_processor_text

        result = detect_multimodal_model("unknown/text-model")
        assert result == False

    @patch("transformers.AutoProcessor")
    def test_processor_loading_failure_graceful(self, mock_auto_processor):
        """Test graceful handling of processor loading failures."""
        mock_auto_processor.from_pretrained.side_effect = Exception("Loading failed")

        # Should not crash and return False
        result = detect_multimodal_model("unknown/model")
        assert result == False


class TestAutoProcessorLoading:
    """Test suite for automatic processor loading."""

    @patch("transformers.AutoProcessor")
    def test_successful_processor_loading(self, mock_auto_processor):
        """Test successful AutoProcessor loading."""
        mock_processor = Mock()
        mock_auto_processor.from_pretrained.return_value = mock_processor

        result = auto_load_processor("Qwen/Qwen2.5-VL-3B-Instruct")

        assert result == mock_processor
        mock_auto_processor.from_pretrained.assert_called_once_with(
            "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True
        )

    @patch("transformers.AutoProcessor")
    def test_processor_loading_failure(self, mock_auto_processor):
        """Test handling of processor loading failures."""
        mock_auto_processor.from_pretrained.side_effect = Exception("Model not found")

        result = auto_load_processor("nonexistent/model")

        assert result is None


class TestEmotionDataTypeDetection:
    """Test suite for emotion data type detection."""

    @pytest.fixture
    def temp_text_data_dir(self):
        """Create temporary directory with text emotion data."""
        temp_dir = Path(tempfile.mkdtemp())

        emotions = ["anger", "happiness", "sadness"]
        for emotion in emotions:
            scenarios = [
                f"You feel {emotion} when something bad happens",
                f"A typical {emotion} scenario would be this situation",
                f"This makes you experience {emotion} strongly",
            ]
            with open(temp_dir / f"{emotion}.json", "w") as f:
                json.dump(scenarios, f)

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_image_data_dir(self):
        """Create temporary directory with image path data."""
        temp_dir = Path(tempfile.mkdtemp())

        emotions = ["anger", "happiness", "sadness"]
        for emotion in emotions:
            image_paths = [
                f"IAPS/Images/{emotion}_001.jpg",
                f"IAPS/Images/{emotion}_002.jpg",
                f"dataset/emotions/{emotion}/image1.png",
            ]
            with open(temp_dir / f"{emotion}.json", "w") as f:
                json.dump(image_paths, f)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_detect_text_data(self, temp_text_data_dir):
        """Test detection of text emotion data."""
        result = detect_emotion_data_type(temp_text_data_dir)

        assert result["data_type"] == "text"
        assert result["is_multimodal_data"] == False
        assert len(result["available_emotions"]) == 3
        assert "anger" in result["available_emotions"]
        assert result["valid_emotions_count"] == 3

    def test_detect_image_data(self, temp_image_data_dir):
        """Test detection of image path data."""
        result = detect_emotion_data_type(temp_image_data_dir)

        assert result["data_type"] == "image"
        assert result["is_multimodal_data"] == True
        assert len(result["available_emotions"]) == 3
        assert "anger" in result["available_emotions"]
        assert result["valid_emotions_count"] == 3

    def test_detect_missing_directory(self):
        """Test handling of missing data directory."""
        result = detect_emotion_data_type("/nonexistent/path")

        assert result["data_type"] == "none"
        assert result["is_multimodal_data"] == False
        assert len(result["available_emotions"]) == 0
        assert result["valid_emotions_count"] == 0

    def test_detect_mixed_data(self, temp_text_data_dir):
        """Test detection when some emotions have text, others missing."""
        # Remove one emotion file to create mixed situation
        (temp_text_data_dir / "sadness.json").unlink()

        result = detect_emotion_data_type(temp_text_data_dir)

        assert result["data_type"] == "text"  # Still text type for available emotions
        assert len(result["available_emotions"]) == 2  # anger, happiness only
        assert result["valid_emotions_count"] == 2


class TestExperimentFeasibilityValidation:
    """Test suite for experiment feasibility validation."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "data_dir": "data/image",
            "rep_token": -1,
            "emotions": ["anger", "happiness", "sadness"],
        }

    @patch("neuro_manipulation.utils.detect_multimodal_model")
    @patch("neuro_manipulation.utils.detect_emotion_data_type")
    def test_explicit_multimodal_feasible(
        self, mock_detect_data, mock_detect_model, base_config
    ):
        """Test explicit multimodal request that is feasible."""
        # Setup mocks
        mock_detect_model.return_value = True
        mock_detect_data.return_value = {
            "data_type": "image",
            "is_multimodal_data": True,
            "valid_emotions_count": 3,
            "available_emotions": ["anger", "happiness", "sadness"],
        }

        base_config["multimodal_intent"] = True

        result = validate_multimodal_experiment_feasibility(base_config)

        assert result["feasible"] == True
        assert result["mode"] == "multimodal"
        assert "explicitly requested and feasible" in " ".join(result["reasons"])

    @patch("neuro_manipulation.utils.detect_multimodal_model")
    @patch("neuro_manipulation.utils.detect_emotion_data_type")
    def test_explicit_multimodal_impossible_text_model(
        self, mock_detect_data, mock_detect_model, base_config
    ):
        """Test explicit multimodal request with text-only model."""
        mock_detect_model.return_value = False  # Text-only model
        mock_detect_data.return_value = {
            "data_type": "image",
            "is_multimodal_data": True,
            "valid_emotions_count": 3,
        }

        base_config["multimodal_intent"] = True

        result = validate_multimodal_experiment_feasibility(base_config)

        assert result["feasible"] == False
        assert result["mode"] == "impossible"
        assert "text-only" in " ".join(result["reasons"])

    @patch("neuro_manipulation.utils.detect_multimodal_model")
    @patch("neuro_manipulation.utils.detect_emotion_data_type")
    def test_auto_detection_suggests_multimodal(
        self, mock_detect_data, mock_detect_model, base_config
    ):
        """Test auto-detection suggesting multimodal but defaulting to text-only."""
        mock_detect_model.return_value = True
        mock_detect_data.return_value = {
            "data_type": "image",
            "is_multimodal_data": True,
            "valid_emotions_count": 3,
            "available_emotions": ["anger", "happiness", "sadness"],
        }

        # No explicit multimodal_intent
        base_config.pop("multimodal_intent", None)

        result = validate_multimodal_experiment_feasibility(base_config)

        assert result["feasible"] == True
        assert result["mode"] == "text_only"  # Defaults to text-only
        assert "Consider setting" in " ".join(result["reasons"])
        assert "multimodal_intent: true" in " ".join(result["reasons"])

    @patch("neuro_manipulation.utils.detect_multimodal_model")
    @patch("neuro_manipulation.utils.detect_emotion_data_type")
    def test_insufficient_emotion_data(
        self, mock_detect_data, mock_detect_model, base_config
    ):
        """Test handling of insufficient emotion data."""
        mock_detect_model.return_value = True
        mock_detect_data.return_value = {
            "data_type": "image",
            "is_multimodal_data": True,
            "valid_emotions_count": 1,  # Less than minimum of 2
            "available_emotions": ["anger"],
        }

        base_config["multimodal_intent"] = True

        result = validate_multimodal_experiment_feasibility(base_config)

        assert result["feasible"] == False
        assert result["mode"] == "impossible"
        assert "only 1 emotions" in " ".join(result["reasons"])


class TestEnhancedDatasetGeneration:
    """Test suite for enhanced primary_emotions_concept_dataset function."""

    @pytest.fixture
    def temp_image_data_dir(self):
        """Create temp directory with image paths."""
        temp_dir = Path(tempfile.mkdtemp())

        emotions = ["anger", "happiness"]
        for emotion in emotions:
            image_paths = [f"IAPS/{emotion}_001.jpg", f"IAPS/{emotion}_002.jpg"]
            with open(temp_dir / f"{emotion}.json", "w") as f:
                json.dump(image_paths, f)

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_text_data_dir(self):
        """Create temp directory with text scenarios."""
        temp_dir = Path(tempfile.mkdtemp())

        emotions_data = {
            "anger": [
                "You discover someone has lied to you",
                "A friend betrays your trust",
            ],
            "happiness": ["You get amazing news", "A surprise visit from loved ones"],
        }

        for emotion, scenarios in emotions_data.items():
            with open(temp_dir / f"{emotion}.json", "w") as f:
                json.dump(scenarios, f)

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_image_base_dir(self):
        """Create actual images for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        iaps_dir = temp_dir / "IAPS"
        iaps_dir.mkdir()

        # Create sample images
        for emotion in ["anger", "happiness"]:
            for i in [1, 2]:
                img = Image.new(
                    "RGB", (100, 100), color="red" if emotion == "anger" else "yellow"
                )
                img.save(iaps_dir / f"{emotion}_{i:03d}.jpg")

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.name_or_path = "test-model"
        return tokenizer

    def test_multimodal_intent_with_image_data(
        self, temp_image_data_dir, temp_image_base_dir, mock_tokenizer
    ):
        """Test multimodal processing with explicit intent and image data."""
        with patch("neuro_manipulation.utils.PromptFormat") as mock_prompt_format:
            mock_format_instance = Mock()
            mock_format_instance.build.return_value = "formatted_prompt"
            mock_prompt_format.return_value = mock_format_instance

            data = primary_emotions_concept_dataset(
                str(temp_image_data_dir),
                model_name="test-model",
                tokenizer=mock_tokenizer,
                multimodal_intent=True,
                image_base_path=str(temp_image_base_dir),
            )

            # Should return multimodal format
            assert "anger" in data
            train_data = data["anger"]["train"]["data"]
            assert len(train_data) > 0

            # Check multimodal format
            for item in train_data:
                if item is not None:  # Skip failed image loads
                    assert isinstance(item, dict)
                    assert "images" in item
                    assert "text" in item
                    assert len(item["images"]) == 1
                    assert isinstance(item["images"][0], Image.Image)

    def test_no_multimodal_intent_with_image_data(
        self, temp_image_data_dir, temp_image_base_dir, mock_tokenizer
    ):
        """Test that multimodal_intent=False uses text-only mode even with image data."""
        with patch("neuro_manipulation.utils.PromptFormat") as mock_prompt_format:
            mock_format_instance = Mock()
            mock_format_instance.build.return_value = "formatted_prompt"
            mock_prompt_format.return_value = mock_format_instance

            data = primary_emotions_concept_dataset(
                str(temp_image_data_dir),
                model_name="test-model",
                tokenizer=mock_tokenizer,
                multimodal_intent=False,  # Explicitly disabled
                image_base_path=str(temp_image_base_dir),
            )

            # Should return text-only format
            train_data = data["anger"]["train"]["data"]
            for item in train_data:
                assert isinstance(item, str), "Should be text-only format"

    def test_direct_data_dir_text_mode(self, temp_text_data_dir):
        """Test function: data_dir points directly to text JSONs."""
        data = primary_emotions_concept_dataset(
            str(temp_text_data_dir),  # Direct path to emotion JSONs
            multimodal_intent=False,
        )

        # Should process available emotions
        assert "anger" in data
        assert "happiness" in data

        # Should be text format
        train_data = data["anger"]["train"]["data"]
        assert len(train_data) > 0
        assert isinstance(train_data[0], str)
        assert "Consider the emotion" in train_data[0]
        assert "lied" in train_data[0] or "betrays" in train_data[0]

    def test_direct_data_dir_image_mode(self):
        """Test function: data_dir points directly to image JSONs with actual images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create actual images in temp_dir (simulating data/image/ directory)
            emotions = ["anger", "happiness"]
            for emotion in emotions:
                # Create test image
                img = Image.new(
                    "RGB", (32, 32), color="red" if emotion == "anger" else "yellow"
                )
                img_path = Path(temp_dir) / f"{emotion}_test.jpg"
                img.save(img_path)

                # Create JSON with relative path
                with open(Path(temp_dir) / f"{emotion}.json", "w") as f:
                    json.dump([f"{emotion}_test.jpg"], f)  # Relative to temp_dir

            data = primary_emotions_concept_dataset(
                temp_dir,  # Direct path containing both JSONs and images
                multimodal_intent=True,
            )

            # Should process emotions
            assert "anger" in data
            assert "happiness" in data

            # Should be multimodal format
            train_data = data["anger"]["train"]["data"]
            assert len(train_data) > 0

            sample = train_data[0]
            assert isinstance(sample, dict)
            assert "images" in sample
            assert "text" in sample
            assert len(sample["images"]) == 1
            assert isinstance(sample["images"][0], Image.Image)
            assert "Consider the emotion" in sample["text"]
            assert "[IMAGE]" in sample["text"]

    def test_auto_detection_behavior(self, temp_image_data_dir):
        """Test function: auto-detection with image data but no multimodal_intent."""
        data = primary_emotions_concept_dataset(
            str(temp_image_data_dir),  # Contains image paths
            # No multimodal_intent specified - should auto-detect but default to text-only
        )

        # Should detect image data but use text-only mode
        assert "anger" in data
        train_data = data["anger"]["train"]["data"]
        assert len(train_data) > 0
        assert isinstance(train_data[0], str)  # Text-only despite image paths
        assert "IAPS" in train_data[0]  # Image path processed as text


class TestLoadModelTokenizerEnhancement:
    """Test suite for enhanced load_model_tokenizer function."""

    @patch("neuro_manipulation.utils.detect_multimodal_model")
    @patch("neuro_manipulation.utils.auto_load_processor")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_auto_load_multimodal_components(
        self,
        mock_tokenizer_cls,
        mock_model_cls,
        mock_auto_load_processor,
        mock_detect_multimodal,
    ):
        """Test automatic loading of multimodal components."""
        # Setup mocks
        mock_detect_multimodal.return_value = True
        mock_processor = Mock()
        mock_auto_load_processor.return_value = mock_processor
        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Test the function
        model, tokenizer, processor = load_model_tokenizer(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )

        # Assertions
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert processor == mock_processor
        mock_detect_multimodal.assert_called_once_with("Qwen/Qwen2.5-VL-3B-Instruct")
        mock_auto_load_processor.assert_called_once_with("Qwen/Qwen2.5-VL-3B-Instruct")

    @patch("neuro_manipulation.utils.detect_multimodal_model")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_text_only_model_no_processor(
        self, mock_tokenizer_cls, mock_model_cls, mock_detect_multimodal
    ):
        """Test that text-only models don't load processor."""
        mock_detect_multimodal.return_value = False
        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        model, tokenizer, processor = load_model_tokenizer("Qwen/Qwen2.5-3B-Instruct")

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert processor is None  # No processor for text-only models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
