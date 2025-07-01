#!/usr/bin/env python3
"""
OpenAI-Compatible Server for RepControlVLLMHook

This script creates an OpenAI-compatible server that integrates RepControlVLLMHook
to enable emotion-controlled language model generation through standard OpenAI client interfaces.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from constants import Emotions  # noqa: E402
from neuro_manipulation.configs.experiment_config import get_repe_eng_config  # noqa: E402
from neuro_manipulation.model_layer_detector import ModelLayerDetector  # noqa: E402
from neuro_manipulation.model_utils import load_emotion_readers  # noqa: E402
from neuro_manipulation.repe.pipelines import repe_pipeline_registry  # noqa: E402
from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook  # noqa: E402
from neuro_manipulation.utils import load_model_tokenizer  # noqa: E402
from vllm import LLM  # noqa: E402

repe_pipeline_registry()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global server state
server_state = {
    "model": None,
    "tokenizer": None,
    "rep_control_hook": None,
    "emotion_readers": None,
    "control_layers": None,
    "model_name": None,
    "current_emotion": None,
    "server_start_time": None,
}


# OpenAI API Models
class ChatMessage(BaseModel):
    role: Optional[str] = Field(default="user", description="The role of the message author")
    content: str = Field(..., description="The contents of the message")

    # Allow additional fields for LangGraph compatibility
    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="The messages in the conversation")
    max_tokens: Optional[int] = Field(
        default=100, description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(default=1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(default=1, description="Number of completions to generate")
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream back partial progress"
    )
    stop: Optional[List[str]] = Field(
        default=None, description="Up to 4 sequences where generation will stop"
    )


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]


# FastAPI app
app = FastAPI(
    title="RepControlVLLMHook OpenAI Server",
    description="OpenAI-compatible API server with emotion control via RepControlVLLMHook",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_emotion_activation_vectors(model_path: str, emotion: str) -> Dict[int, torch.Tensor]:
    """Load emotion activation vectors for the given emotion."""
    logger.info(f"Loading emotion activation vectors for '{emotion}'...")

    # Configuration for emotion readers (matching the existing experiment setup)
    config = get_repe_eng_config(model_path)

    # Load temporary model for reading vectors
    logger.info("Loading temporary model for emotion vector extraction...")
    temp_model, temp_tokenizer = load_model_tokenizer(
        model_path, expand_vocab=False, from_vllm=False
    )

    # Get layer information
    num_layers = ModelLayerDetector.num_layers(temp_model)
    hidden_layers = list(range(-1, -num_layers - 1, -1))
    control_layers = hidden_layers[len(hidden_layers) // 3:2 * len(hidden_layers) // 3]

    # Load emotion readers
    emotion_readers = load_emotion_readers(config, temp_model, temp_tokenizer, hidden_layers)

    # Clean up temporary model
    del temp_model
    del temp_tokenizer
    torch.cuda.empty_cache()

    # Extract activation vectors for the specific emotion
    if emotion not in emotion_readers:
        raise ValueError(
            f"Emotion '{emotion}' not found in loaded readers. "
            f"Available: {list(emotion_readers.keys())}"
        )

    rep_reader = emotion_readers[emotion]
    activation_intensity = 1.5  # Default intensity

    activations = {
        layer: torch.tensor(
            activation_intensity * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
        )
        .cpu()
        .half()
        for layer in control_layers
    }

    logger.info(f"Loaded activation vectors for {len(control_layers)} layers")
    return activations, control_layers, emotion_readers


def initialize_model(model_path: str, emotion: str, tensor_parallel_size: int = 1):
    """Initialize the vLLM model with RepControlVLLMHook."""
    logger.info(f"Initializing model: {model_path}")
    logger.info(f"Target emotion: {emotion}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load emotion activation vectors
    activations, control_layers, emotion_readers = load_emotion_activation_vectors(
        model_path, emotion
    )

    # Initialize vLLM model
    logger.info("Initializing vLLM...")
    model = LLM(
        model=model_path,
        tokenizer=model_path,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_num_seqs=16,
    )

    # Initialize RepControlVLLMHook
    logger.info("Initializing RepControlVLLMHook...")
    rep_control = RepControlVLLMHook(
        model=model,
        tokenizer=tokenizer,
        layers=control_layers,
        block_name="decoder_block",
        control_method="reading_vec",
        tensor_parallel_size=tensor_parallel_size,
    )

    # Store in global state
    server_state["model"] = model
    server_state["tokenizer"] = tokenizer
    server_state["rep_control_hook"] = rep_control
    server_state["emotion_readers"] = emotion_readers
    server_state["control_layers"] = control_layers
    server_state["current_emotion"] = emotion
    server_state["emotion_activations"] = activations

    logger.info("Model initialization complete")


def apply_emotion_to_prompt(messages: List[ChatMessage], emotion: str) -> str:
    """Convert OpenAI chat messages to a single prompt string."""
    # Simple conversion - in practice this could be more sophisticated
    prompt_parts = []
    for message in messages:
        # Handle LangGraph format (messages without role)
        role = getattr(message, "role", "user")
        content = getattr(message, "content", str(message))

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            # Default to user if role is unknown
            prompt_parts.append(f"User: {content}")

    prompt = "\n".join(prompt_parts) + "\nAssistant:"
    return prompt


async def stream_chat_completion(
    completion_id: str,
    created_time: int,
    model: str,
    full_text: str,
    prompt_tokens: int,
    completion_tokens: int,
):
    """Stream a chat completion response in OpenAI format."""
    # Split text into chunks for streaming
    words = full_text.split()
    chunk_size = 2  # Words per chunk

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk_text += " "

        chunk_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }

        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.05)  # Small delay between chunks

    # Send final chunk with finish_reason
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    if server_state["model_name"] is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    return ModelsResponse(
        data=[
            Model(
                id=server_state["model_name"],
                created=int(server_state["server_start_time"]),
                owned_by="local",
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion with emotion control."""
    if server_state["rep_control_hook"] is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Convert messages to prompt (handle LangGraph format)
        prompt = apply_emotion_to_prompt(request.messages, server_state["current_emotion"])

        # Prepare sampling parameters (not used directly with hook)
        # sampling_params = SamplingParams(
        #     max_tokens=request.max_tokens or 100,
        #     temperature=request.temperature or 0.0,
        #     top_p=request.top_p or 1.0,
        #     stop=request.stop,
        # )

        # Generate with emotion control
        logger.info(f"Generating with emotion: {server_state['current_emotion']}")
        outputs = server_state["rep_control_hook"](
            text_inputs=[prompt],
            activations=server_state["emotion_activations"],
            max_new_tokens=request.max_tokens or 100,
            temperature=request.temperature or 0.0,
            top_p=request.top_p or 1.0,
            operator="linear_comb",
            normalize=False,
            token_pos=None,
        )

        # Extract generated text
        generated_text = outputs[0].outputs[0].text.strip()

        # Count tokens (approximate)
        prompt_tokens = len(server_state["tokenizer"].encode(prompt))
        completion_tokens = len(server_state["tokenizer"].encode(generated_text))

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(
                    completion_id,
                    created_time,
                    request.model,
                    generated_text,
                    prompt_tokens,
                    completion_tokens,
                ),
                media_type="text/plain",
            )
        else:
            # Create response
            response = ChatCompletionResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generated_text),
                        finish_reason="stop",
                    )
                ],
                usage=ChatUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

            return response

    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": server_state["model"] is not None,
        "current_emotion": server_state["current_emotion"],
        "server_time": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for RepControlVLLMHook")
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--model_name", required=True, help="Model name for API")
    parser.add_argument("--emotion", default="anger", help="Emotion to activate (default: anger)")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--api_key", help="API key for authentication (optional)")
    parser.add_argument("--url", help="Base URL for the server (for display purposes)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")

    args = parser.parse_args()

    # Validate emotion
    valid_emotions = Emotions.get_emotions()
    if args.emotion not in valid_emotions:
        logger.error(f"Invalid emotion '{args.emotion}'. Valid emotions: {valid_emotions}")
        sys.exit(1)

    # Set server start time
    server_state["server_start_time"] = time.time()
    server_state["model_name"] = args.model_name

    # Initialize model
    try:
        initialize_model(args.model, args.emotion, args.tensor_parallel_size)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        sys.exit(1)

    # Display server info
    logger.info("=" * 60)
    logger.info("RepControlVLLMHook OpenAI Server")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Emotion: {args.emotion}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    if args.url:
        logger.info(f"Base URL: {args.url}")
    if args.api_key:
        logger.info(f"API Key: {args.api_key}")
    logger.info("=" * 60)

    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
