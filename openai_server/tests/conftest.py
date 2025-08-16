"""
Pytest configuration for OpenAI Server tests

This file contains shared fixtures and configuration for all tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_model_path():
    """Provide a mock model path for testing"""
    return "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def test_port():
    """Provide a test port number"""
    return 8765


@pytest.fixture
def test_emotions():
    """Provide test emotion list"""
    return ["anger", "happiness", "sadness", "fear", "surprise", "disgust"]


@pytest.fixture
def sample_chat_messages():
    """Provide sample chat messages for testing"""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


@pytest.fixture
def mock_vllm_engine(monkeypatch):
    """Mock vLLM engine for unit tests"""
    from unittest.mock import MagicMock, Mock

    mock_engine = Mock()
    mock_engine.generate = MagicMock()

    # Mock the LLM class
    mock_llm_class = Mock()
    mock_llm_class.return_value = mock_engine

    monkeypatch.setattr("openai_server.server.LLM", mock_llm_class)

    return mock_engine


@pytest.fixture
def mock_tokenizer(monkeypatch):
    """Mock tokenizer for unit tests"""
    from unittest.mock import Mock

    mock_tok = Mock()
    mock_tok.encode = Mock(return_value=[1, 2, 3, 4, 5])
    mock_tok.decode = Mock(return_value="Test response")
    mock_tok.apply_chat_template = Mock(return_value="Formatted prompt")

    return mock_tok


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available resources"""
    import torch

    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
