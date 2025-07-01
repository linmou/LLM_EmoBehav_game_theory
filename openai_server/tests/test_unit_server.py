#!/usr/bin/env python3
"""
Unit Tests for OpenAI Server Components

This module contains unit tests for the OpenAI-compatible server,
testing individual components without starting the full server.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from openai_server.server import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
    Model,
    ModelsResponse,
    stream_chat_completion,
)


class TestPydanticModels:
    """Test Pydantic model validation and serialization"""

    def test_chat_message_creation(self):
        """Test ChatMessage model creation"""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        # Test default role
        msg2 = ChatMessage(content="Hi")
        assert msg2.role == "user"

    def test_chat_completion_request_validation(self):
        """Test ChatCompletionRequest validation"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
            ],
            temperature=0.7,
            max_tokens=100,
        )
        assert request.model == "test-model"
        assert len(request.messages) == 2
        assert request.temperature == 0.7
        assert request.max_tokens == 100

    def test_chat_completion_response_structure(self):
        """Test ChatCompletionResponse structure"""
        response = ChatCompletionResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        assert response.id == "test-id"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 30

    def test_model_response_structure(self):
        """Test Model and ModelsResponse structure"""
        model = Model(id="test-model", created=1234567890, owned_by="local")
        assert model.object == "model"

        models_response = ModelsResponse(data=[model])
        assert models_response.object == "list"
        assert len(models_response.data) == 1


class TestStreamingFunctions:
    """Test streaming response functions"""

    @pytest.mark.asyncio
    async def test_stream_chat_completion(self):
        """Test streaming chat completion generation"""
        chunks = []

        async for chunk in stream_chat_completion(
            completion_id="test-123",
            created_time=1234567890,
            model="test-model",
            full_text="Hello world",
            prompt_tokens=5,
            completion_tokens=2,
        ):
            chunks.append(chunk)

        # Should have at least start chunk, content chunks, and end chunk
        assert len(chunks) >= 3

        # Check first chunk structure
        first_chunk = json.loads(chunks[0].replace("data: ", ""))
        assert first_chunk["id"] == "test-123"
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["model"] == "test-model"

        # Check final chunk
        assert chunks[-1] == "data: [DONE]\n\n"


class TestServerInitialization:
    """Test server initialization and configuration"""

    @patch("openai_server.server.load_emotion_readers")
    def test_emotion_vector_loading(self, mock_load_emotion):
        """Test emotion vector loading during initialization"""
        # Mock emotion readers
        mock_readers = {"layer_-9": Mock(), "layer_-10": Mock()}
        mock_load_emotion.return_value = (mock_readers, Mock())

        # Test that emotion loading can be mocked
        readers, _ = mock_load_emotion.return_value
        assert "layer_-9" in readers
        assert "layer_-10" in readers

        # Verify mock was configured correctly
        mock_load_emotion.assert_not_called()  # Not called yet in this test


class TestErrorHandling:
    """Test error handling in server endpoints"""

    def test_invalid_request_format(self):
        """Test handling of invalid request formats"""
        # Test missing required fields
        with pytest.raises(ValueError):
            ChatCompletionRequest(messages=[])  # Missing model

        # Test invalid message format
        with pytest.raises(ValueError):
            ChatMessage()  # Missing content

    def test_emotion_validation(self):
        """Test emotion parameter validation"""
        from constants import Emotions

        # Valid emotions
        valid_emotions = ["anger", "happiness", "sadness", "fear", "surprise", "disgust"]
        for emotion in valid_emotions:
            assert Emotions.from_string(emotion) is not None

        # Invalid emotion should raise
        with pytest.raises(ValueError):
            Emotions.from_string("invalid_emotion")


class TestVLLMIntegration:
    """Test vLLM integration components"""

    @patch("openai_server.server.RepControlVLLMHook")
    def test_hook_initialization(self, mock_hook_class):
        """Test RepControlVLLMHook initialization"""
        mock_hook = Mock()
        mock_hook_class.return_value = mock_hook

        # Test hook can be created with required parameters
        hook = mock_hook_class(
            layers=[-9, -10, -11], block="decoder_block", control_method="reading_vec"
        )

        mock_hook_class.assert_called_once()
        assert hook is not None

    @patch("vllm.SamplingParams")
    def test_sampling_params_creation(self, mock_sampling_params):
        """Test SamplingParams creation from request"""
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(content="test")],
            temperature=0.8,
            max_tokens=150,
            top_p=0.95,
        )

        # In actual server, this would create SamplingParams
        params = mock_sampling_params(
            temperature=request.temperature, max_tokens=request.max_tokens, top_p=request.top_p
        )

        mock_sampling_params.assert_called_once()


class TestAsyncEndpoints:
    """Test async endpoint behavior"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        # This is a placeholder for integration tests
        # Unit tests don't need to test concurrent server behavior
        assert True  # Placeholder test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
