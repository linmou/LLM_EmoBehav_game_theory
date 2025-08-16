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
import re
import sys
import time
import uuid
from collections import deque
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
from neuro_manipulation.configs.experiment_config import (  # noqa: E402
    get_repe_eng_config,
)
from neuro_manipulation.model_layer_detector import ModelLayerDetector  # noqa: E402
from neuro_manipulation.model_utils import load_emotion_readers  # noqa: E402
from neuro_manipulation.repe.pipelines import repe_pipeline_registry  # noqa: E402
from neuro_manipulation.repe.rep_control_vllm_hook import (  # noqa: E402
    RepControlVLLMHook,
)
from neuro_manipulation.utils import load_model_tokenizer  # noqa: E402
from vllm import LLM  # noqa: E402

# Import new graceful degradation components
from async_vllm_wrapper import (  # noqa: E402
    initialize_async_vllm_wrapper,
    get_async_vllm_wrapper,
)
from request_queue_manager import (  # noqa: E402
    initialize_request_queue_manager,
    get_request_queue_manager,
    RequestPriority,
)

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
    "batch_queue": None,
    "batch_processor": None,
}


# Function/Tool calling models
class FunctionDefinition(BaseModel):
    name: str = Field(..., description="Function name")
    description: Optional[str] = Field(default="", description="Function description")
    parameters: Dict = Field(
        default_factory=dict, description="JSON schema for function parameters"
    )


class ToolDefinition(BaseModel):
    type: str = Field(default="function", description="Tool type (currently only 'function')")
    function: FunctionDefinition = Field(..., description="Function definition")


class ToolCall(BaseModel):
    id: str = Field(..., description="Unique identifier for the tool call")
    type: str = Field(default="function", description="Tool type")
    function: Dict[str, str] = Field(..., description="Function name and arguments")


# OpenAI API Models
class ChatMessage(BaseModel):
    role: Optional[str] = Field(default="user", description="The role of the message author")
    content: Optional[str] = Field(default="", description="The contents of the message")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls made by assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of tool call this message responds to"
    )

    # Allow additional fields for LangGraph compatibility
    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="The messages in the conversation")
    max_tokens: Optional[int] = Field(
        default=40000, description="Maximum number of tokens to generate"
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
    tools: Optional[List[ToolDefinition]] = Field(
        default=None, description="Available tools/functions the model can call"
    )
    tool_choice: Optional[str] = Field(
        default="auto", description="Control tool usage: 'none', 'auto', or specific function name"
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

# Startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize queue manager on startup."""
    queue_manager = get_request_queue_manager()
    if queue_manager:
        await queue_manager.start()
        logger.info("Request queue manager started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of all components."""
    # Stop queue manager
    queue_manager = get_request_queue_manager()
    if queue_manager:
        await queue_manager.stop()
        logger.info("Request queue manager stopped")
    
    # Shutdown async wrapper
    async_wrapper = get_async_vllm_wrapper()
    if async_wrapper:
        await async_wrapper.shutdown()
        logger.info("Async vLLM wrapper shutdown")


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
    control_layers = hidden_layers[len(hidden_layers) // 3 : 2 * len(hidden_layers) // 3]

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


def initialize_model(
    model_path: str,
    emotion: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 64,
    enable_chunked_prefill: bool = True,
    batch_size: int = 8,
    batch_timeout: float = 0.05,
    disable_batching: bool = False,
    request_timeout: int = 60,
    max_queue_size: int = 50,
    max_concurrent_requests: int = 3,
    queue_rejection_threshold: float = 0.8,
    reset_interval: int = 300,
    vllm_rejection_threshold: float = 0.7,
):
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

    # Initialize vLLM model with configurable settings for optimal GPU utilization
    logger.info("Initializing vLLM...")
    logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
    logger.info(f"Max sequences: {max_num_seqs}")
    logger.info(f"Chunked prefill: {enable_chunked_prefill}")

    # Prepare vLLM arguments with only supported parameters
    vllm_args = {
        "model": model_path,
        "tokenizer": model_path,
        "enforce_eager": True,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_seqs * 512,
    }

    # Add optional parameters if supported
    if enable_chunked_prefill:
        vllm_args["enable_chunked_prefill"] = True

    # Check if disable_log_requests is supported (might be disable_log_stats)
    try:
        vllm_args["disable_log_stats"] = True
    except:
        pass

    model = LLM(**vllm_args)

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
    server_state["request_timeout"] = request_timeout  # Store timeout for later use

    # Initialize batch processor for better throughput (if not disabled)
    if not disable_batching:
        server_state["batch_processor"] = BatchProcessor(
            max_batch_size=batch_size, batch_timeout=batch_timeout
        )
        logger.info(
            f"Batch processor enabled: max_batch_size={batch_size}, timeout={batch_timeout}s"
        )
    else:
        server_state["batch_processor"] = None
        logger.info("Batch processing disabled")

    # Initialize health monitoring
    try:
        from health_monitor import get_health_monitor
        health_monitor = get_health_monitor()
        health_monitor.start_monitoring()
        logger.info("Health monitoring started")
    except Exception as e:
        logger.warning(f"Failed to start health monitoring: {e}")
    
    # Initialize AsyncVLLMWrapper for timeout protection
    try:
        async_wrapper = initialize_async_vllm_wrapper(
            vllm_hook=rep_control,
            default_timeout=request_timeout,
            max_workers=max_concurrent_requests,  # Match queue capacity
            reset_interval=reset_interval,
            rejection_start_threshold=vllm_rejection_threshold
        )
        logger.info(
            f"AsyncVLLMWrapper initialized with {request_timeout}s timeout, "
            f"{max_concurrent_requests} max workers, {reset_interval}s reset interval, "
            f"{vllm_rejection_threshold} rejection threshold"
        )
    except Exception as e:
        logger.error(f"Failed to initialize AsyncVLLMWrapper: {e}")
        raise
    
    # Initialize RequestQueueManager for load management
    try:
        queue_manager = initialize_request_queue_manager(
            max_queue_size=max_queue_size,
            max_concurrent_requests=max_concurrent_requests,
            queue_timeout=300.0,  # Keep 5 minutes for queue timeout
            rejection_threshold=queue_rejection_threshold
        )
        logger.info(f"RequestQueueManager initialized: max_queue={max_queue_size}, max_concurrent={max_concurrent_requests}, rejection_threshold={queue_rejection_threshold}")
    except Exception as e:
        logger.error(f"Failed to initialize RequestQueueManager: {e}")
        raise

    logger.info("Model initialization complete")


def format_tools_for_prompt(tools: Optional[List[ToolDefinition]]) -> str:
    """Format tools as JSON schema for the model."""
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        func = tool.function
        tool_desc = {
            "name": func.name,
            "description": func.description,
            "parameters": func.parameters,
        }
        tool_descriptions.append(tool_desc)

    tools_text = json.dumps(tool_descriptions, indent=2)
    
    # Enhanced prompt for better SWE-agent compatibility
    return f"""You have access to the following tools/functions:

{tools_text}

To use a tool, you can use ANY of these formats:

1. JSON format (preferred):
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "<function_name>", "arguments": "<json_string>"}}}}]}}

2. For bash commands, you can use markdown code blocks:
```bash
your command here
```

3. Natural language with clear intent:
- "Run bash command: ls -la"
- "Execute: cd /path && python script.py"
- "Read file: /path/to/file.py"

Important for SWE-agent tasks:
- Use 'bash' tool for running commands
- Code blocks with ```bash or ``` are automatically treated as bash commands
- Multiple commands can be run in sequence
- Be explicit about which tool you're using"""


def parse_tool_calls_from_response(response_text: str) -> List[ToolCall]:
    """Parse tool calls from model response."""
    tool_calls = []

    try:
        # Method 1: Try to parse the entire response as JSON
        try:
            data = json.loads(response_text)
            if "tool_calls" in data:
                for call_data in data["tool_calls"]:
                    tool_call = ToolCall(
                        id=call_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        type=call_data.get("type", "function"),
                        function={
                            "name": call_data["function"]["name"],
                            "arguments": call_data["function"]["arguments"],
                        },
                    )
                    tool_calls.append(tool_call)
                return tool_calls
        except json.JSONDecodeError:
            pass

        # Method 2: Look for JSON objects containing tool_calls with more flexible pattern
        # Use a pattern that can handle nested braces
        brace_count = 0
        start_idx = -1

        for i, char in enumerate(response_text):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Found a complete JSON object
                    json_str = response_text[start_idx : i + 1]
                    try:
                        data = json.loads(json_str)
                        if "tool_calls" in data:
                            for call_data in data["tool_calls"]:
                                # Ensure arguments is a JSON string
                                arguments = call_data["function"]["arguments"]
                                if isinstance(arguments, dict):
                                    arguments = json.dumps(arguments)
                                elif not isinstance(arguments, str):
                                    arguments = json.dumps({"argument": str(arguments)})
                                    
                                tool_call = ToolCall(
                                    id=call_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                    type=call_data.get("type", "function"),
                                    function={
                                        "name": call_data["function"]["name"],
                                        "arguments": arguments,
                                    },
                                )
                                tool_calls.append(tool_call)
                            return tool_calls
                    except json.JSONDecodeError:
                        continue
                    start_idx = -1
        
        # Method 3: Enhanced detection for SWE-agent tools in markdown code blocks
        # This is crucial for SWE-agent compatibility
        
        # Check for JSON code blocks with tool calls
        json_code_block_pattern = r'```(?:json)?\s*\n(\{[^`]*"tool_calls"[^`]*\})\s*\n```'
        json_matches = re.findall(json_code_block_pattern, response_text, re.DOTALL | re.MULTILINE)
        
        for json_content in json_matches:
            try:
                data = json.loads(json_content)
                if "tool_calls" in data:
                    for call_data in data["tool_calls"]:
                        # Ensure arguments is a JSON string
                        arguments = call_data["function"]["arguments"]
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments)
                        elif not isinstance(arguments, str):
                            arguments = json.dumps({"argument": str(arguments)})
                            
                        tool_call = ToolCall(
                            id=call_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            type=call_data.get("type", "function"),
                            function={
                                "name": call_data["function"]["name"],
                                "arguments": arguments,
                            },
                        )
                        tool_calls.append(tool_call)
                    return tool_calls  # Return early if we found JSON tool calls
            except json.JSONDecodeError:
                continue
        
        # Check for bash/shell code blocks  
        bash_code_block_pattern = r'```(?:bash|sh|shell)?\n(.*?)\n```'
        bash_matches = re.findall(bash_code_block_pattern, response_text, re.DOTALL | re.MULTILINE)
        
        for code_content in bash_matches:
            # Treat code blocks as bash commands
            if code_content.strip():
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function={
                        "name": "bash",
                        "arguments": json.dumps({"command": code_content.strip()})
                    }
                )
                tool_calls.append(tool_call)
        
        # Method 4: Look for SWE-agent specific tool patterns
        if not tool_calls:
            # Common SWE-agent tools
            swe_tools = {
                'bash': r'(?:run|execute|bash)(?:\s+command)?:\s*(.+)',
                'edit_file': r'edit_file\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)',
                'read_file': r'read[_\s]file\s*\(\s*["\']?([^"\']+)["\']?\s*\)',
                'write_file': r'write[_\s]file\s*\(\s*["\']?([^"\']+)["\']?\s*,\s*["\']?([^"\']+)["\']?\s*\)',
            }
            
            for tool_name, pattern in swe_tools.items():
                matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if tool_name == 'bash':
                        arguments = json.dumps({"command": match.strip()})
                    elif tool_name == 'edit_file' and isinstance(match, tuple) and len(match) == 3:
                        arguments = json.dumps({
                            "path": match[0].strip(),
                            "old_str": match[1].strip(),
                            "new_str": match[2].strip()
                        })
                    elif tool_name in ['read_file', 'write_file']:
                        if isinstance(match, tuple):
                            arguments = json.dumps({"path": match[0].strip()})
                        else:
                            arguments = json.dumps({"path": match.strip()})
                    else:
                        continue
                    
                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function={"name": tool_name, "arguments": arguments}
                    )
                    tool_calls.append(tool_call)

        # Method 5: Look for function calls in simpler format (fallback)
        if not tool_calls:
            func_pattern = r"(\w+)\s*\(\s*([^)]*)\s*\)"
            func_matches = re.findall(func_pattern, response_text)

            for func_name, args_str in func_matches:
                try:
                    # Try to parse arguments as JSON
                    if args_str.strip():
                        # If it looks like JSON, validate it
                        if args_str.strip().startswith("{"):
                            arguments = json.dumps(json.loads(args_str))
                        else:
                            # Simple string argument, wrap in JSON
                            arguments = json.dumps({"argument": args_str})
                    else:
                        arguments = "{}"

                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function={"name": func_name, "arguments": arguments},
                    )
                    tool_calls.append(tool_call)
                except:
                    continue

    except Exception as e:
        logger.warning(f"Failed to parse tool calls: {e}")

    return tool_calls


def clean_content_from_tool_calls(response_text: str) -> str:
    """Remove tool call JSON from content while preserving other text."""
    content = response_text
    
    # Remove JSON code blocks containing tool calls (improved pattern)
    json_code_block_pattern = r'```(?:json)?\s*\n\{[^`]*"tool_calls"[^`]*\}\s*\n```'
    content = re.sub(json_code_block_pattern, '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Remove markdown JSON blocks that don't have proper closing (improved)
    json_block_unclosed_pattern = r'```json\s*\n\{[^`]*"tool_calls"[^`]*(?:\]\s*)?(?:\}\s*)?[^`]*'
    content = re.sub(json_block_unclosed_pattern, '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Remove inline JSON with tool calls (improved pattern using balanced braces)
    def remove_tool_call_json(text):
        # Handle malformed JSON at start of text
        if text.strip().startswith('{"tool_calls"'):
            # Find the end of the JSON object or take everything until newline
            brace_count = 0
            i = 0
            for char in text:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON, remove it and keep rest
                        return text[i+1:].lstrip('\n')
                elif char == '\n' and brace_count > 0:
                    # Incomplete JSON until newline, remove just that line
                    return text[i+1:]
                i += 1
            # If no complete JSON found, remove the whole line
            newline_pos = text.find('\n')
            if newline_pos != -1:
                return text[newline_pos+1:]
            else:
                return ""
        
        # Find JSON objects that contain "tool_calls"
        brace_count = 0
        start_pos = -1
        i = 0
        while i < len(text):
            if text[i] == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    # Check if this JSON contains tool_calls
                    json_str = text[start_pos:i+1]
                    if '"tool_calls"' in json_str:
                        # Remove this JSON object and any trailing text until newline
                        end_pos = text.find('\n', i+1)
                        if end_pos == -1:
                            end_pos = len(text)
                        text = text[:start_pos] + text[end_pos:]
                        i = start_pos - 1  # Reset position
                    start_pos = -1
            i += 1
        return text
    
    content = remove_tool_call_json(content)
    
    # Remove any remaining standalone tool_calls patterns
    content = re.sub(r'(?:^|\n)\s*"tool_calls"\s*:\s*\[.*?\]', '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Remove lines that only contain partial JSON starting with {"tool_calls"
    content = re.sub(r'^.*\{"tool_calls".*$', '', content, flags=re.MULTILINE)
    
    # Clean up whitespace and empty lines
    content = re.sub(r'\n\s*\n', '\n', content)  # Remove empty lines
    content = re.sub(r'^\s*\n', '', content)     # Remove leading empty lines
    content = content.strip()
    
    return content


def apply_emotion_to_prompt(
    messages: List[ChatMessage], emotion: str, tools: Optional[List[ToolDefinition]] = None
) -> str:
    """Convert OpenAI chat messages to a single prompt string with optional tools."""
    prompt_parts = []

    # Add tools information first if available
    if tools:
        tools_prompt = format_tools_for_prompt(tools)
        prompt_parts.append(tools_prompt)

    for message in messages:
        # Handle LangGraph format (messages without role)
        role = getattr(message, "role", "user")
        content = getattr(message, "content", str(message))

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Format assistant message with tool calls
                tool_calls_text = json.dumps(
                    [
                        {"id": call.id, "type": call.type, "function": call.function}
                        for call in message.tool_calls
                    ]
                )
                prompt_parts.append(f"Assistant: {content}\nTool calls: {tool_calls_text}")
            else:
                prompt_parts.append(f"Assistant: {content}")
        elif role == "tool":
            # Tool response message
            tool_id = getattr(message, "tool_call_id", "unknown")
            prompt_parts.append(f"Tool Result (ID: {tool_id}): {content}")
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
        chunk_words = words[i : i + chunk_size]
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


class BatchProcessor:
    """Batches multiple requests for efficient GPU utilization."""

    def __init__(self, max_batch_size: int = 8, batch_timeout: float = 0.05):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue = deque()
        self.processing = False

    async def process_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Add request to batch and wait for response."""
        future = asyncio.Future()
        request_item = {"request": request, "future": future, "timestamp": time.time()}

        self.queue.append(request_item)

        # Start batch processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process accumulated requests in batches."""
        if self.processing:
            return

        self.processing = True

        try:
            while self.queue:
                # Collect batch
                batch = []
                start_time = time.time()

                while (
                    len(batch) < self.max_batch_size
                    and self.queue
                    and (time.time() - start_time) < self.batch_timeout
                ):
                    batch.append(self.queue.popleft())
                    if not self.queue:
                        await asyncio.sleep(0.001)  # Small delay to accumulate more requests

                if batch:
                    await self._process_batch_items(batch)

        finally:
            self.processing = False

    async def _process_batch_items(self, batch):
        """Process a batch of requests together."""
        try:
            # Get async wrapper
            async_wrapper = get_async_vllm_wrapper()
            if not async_wrapper:
                raise HTTPException(status_code=500, detail="AsyncVLLMWrapper not initialized")
            
            # Extract prompts and requests
            prompts = []
            requests = []
            futures = []

            for item in batch:
                request = item["request"]
                future = item["future"]

                prompt = apply_emotion_to_prompt(
                    request.messages, server_state["current_emotion"], request.tools
                )
                prompts.append(prompt)
                requests.append(request)
                futures.append(future)

            # Batch generation
            if len(prompts) > 1:
                logger.debug(f"Processing batch of {len(prompts)} requests")

                # Use batch processing for multiple requests with timeout protection
                outputs = await async_wrapper.generate_async(
                    text_inputs=prompts,
                    activations=server_state["emotion_activations"],
                    max_new_tokens=max(req.max_tokens or 40000 for req in requests),
                    temperature=requests[0].temperature or 0.0,
                    top_p=requests[0].top_p or 1.0,
                    timeout=server_state["request_timeout"] * 1.5,  # Slightly longer timeout for batch
                    operator="linear_comb",
                    normalize=False,
                    token_pos=None,
                )
            else:
                # Single request fallback
                request = requests[0]
                prompt = prompts[0]

                outputs = await async_wrapper.generate_async(
                    text_inputs=[prompt],
                    activations=server_state["emotion_activations"],
                    max_new_tokens=request.max_tokens or 40000,
                    temperature=request.temperature or 0.0,
                    top_p=request.top_p or 1.0,
                    timeout=server_state["request_timeout"],  # Use configured timeout
                    operator="linear_comb",
                    normalize=False,
                    token_pos=None,
                )

            # Process responses
            for i, (request, future) in enumerate(zip(requests, futures)):
                try:
                    generated_text = outputs[i].outputs[0].text.strip()
                    prompt_tokens = len(server_state["tokenizer"].encode(prompts[i]))
                    completion_tokens = len(server_state["tokenizer"].encode(generated_text))

                    # Parse tool calls from generated text
                    tool_calls = None
                    content = generated_text
                    finish_reason = "stop"

                    if request.tools and request.tool_choice != "none":
                        parsed_calls = parse_tool_calls_from_response(generated_text)
                        if parsed_calls:
                            tool_calls = parsed_calls
                            finish_reason = "tool_calls"
                            # Remove tool call JSON from content if it exists
                            content = clean_content_from_tool_calls(generated_text)
                            if not content.strip():
                                content = None

                    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
                    created_time = int(time.time())

                    message = ChatMessage(role="assistant", content=content if content else None)
                    if tool_calls:
                        message.tool_calls = tool_calls

                    response = ChatCompletionResponse(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            ChatChoice(
                                index=0,
                                message=message,
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=ChatUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                    )

                    future.set_result(response)

                except Exception as e:
                    future.set_exception(e)

        except Exception as e:
            # Set exception for all futures in batch
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(e)


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
    """Create a chat completion with emotion control and graceful degradation."""
    if server_state["rep_control_hook"] is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Initialize graceful degradation components
    from circuit_breaker import get_circuit_breaker, CircuitBreakerConfig  
    from adaptive_processor import get_adaptive_processor
    from health_monitor import get_health_monitor
    
    # Get circuit breaker for chat completions
    circuit_breaker = get_circuit_breaker(
        "chat_completions",
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=float(server_state["request_timeout"]),  # Use configured timeout
            health_score_threshold=0.4
        )
    )
    
    # Record request start time for health monitoring
    request_start_time = time.time()
    
    # Process request with adaptive optimization - DISABLED due to ChatMessage type mismatch
    # TODO: Fix ChatMessage type conversion between OpenAI client objects and Pydantic models
    # try:
    #     adaptive_processor = get_adaptive_processor()
    #     optimized_request, processing_info = adaptive_processor.process_request(request)
    #     
    #     # Log optimization details
    #     if processing_info["optimization_applied"]:
    #         logger.info(f"Request optimized: {processing_info['strategy_used']} "
    #                    f"(health: {processing_info['health_score']:.2f})")
    #     
    #     # Use optimized request for processing
    #     request = optimized_request
    #     
    # except HTTPException:
    #     # Re-raise HTTP exceptions (like service unavailable)
    #     raise
    # except Exception as e:
    #     logger.error(f"Request optimization failed: {e}")
    #     # Continue with original request if optimization fails
    processing_info = {"optimization_applied": False, "strategy_used": "disabled"}

    # Check if batching is enabled and request supports it
    use_batching = (
        server_state.get("batch_processor") is not None
        and not request.stream  # Streaming not supported with batching yet
    )

    if use_batching:
        # Use batch processor with circuit breaker protection
        async def batch_process():
            return await server_state["batch_processor"].process_request(request)
        
        try:
            return await circuit_breaker.call_async(batch_process)
        except Exception as e:
            # Record request metrics for health monitoring
            health_monitor = get_health_monitor()
            health_monitor.record_request(time.time() - request_start_time, False)
            raise

    # Fallback to individual processing with circuit breaker protection
    async def individual_process():
        return await _process_individual_request(request, processing_info.get("strategy_used", "unknown"))
    
    try:
        result = await circuit_breaker.call_async(individual_process)
        
        # Record successful request for health monitoring
        health_monitor = get_health_monitor()
        health_monitor.record_request(time.time() - request_start_time, True)
        
        return result
        
    except Exception as e:
        # Record failed request for health monitoring
        health_monitor = get_health_monitor()
        health_monitor.record_request(time.time() - request_start_time, False)
        raise


async def _process_individual_request(request: ChatCompletionRequest, strategy_used: str = "unknown"):
    """Process individual request with emotion control."""
    try:
        # Convert messages to prompt (handle LangGraph format)
        prompt = apply_emotion_to_prompt(
            request.messages, server_state["current_emotion"], request.tools
        )

        # Generate with emotion control - using async wrapper for timeout protection
        logger.debug(f"Generating with emotion: {server_state['current_emotion']}")
        
        # Get async wrapper
        async_wrapper = get_async_vllm_wrapper()
        if not async_wrapper:
            raise HTTPException(status_code=500, detail="AsyncVLLMWrapper not initialized")
        
        # Use async wrapper for timeout protection
        outputs = await async_wrapper.generate_async(
            text_inputs=[prompt],
            activations=server_state["emotion_activations"],
            max_new_tokens=request.max_tokens or 40000,
            temperature=request.temperature or 0.0,
            top_p=request.top_p or 1.0,
            timeout=server_state["request_timeout"],  # Use configured timeout
            operator="linear_comb",
            normalize=False,
            token_pos=None,
        )

        # Extract generated text
        generated_text = outputs[0].outputs[0].text.strip()

        # Count tokens (approximate)
        prompt_tokens = len(server_state["tokenizer"].encode(prompt))
        completion_tokens = len(server_state["tokenizer"].encode(generated_text))

        # Parse tool calls from generated text
        tool_calls = None
        content = generated_text
        finish_reason = "stop"

        if request.tools and request.tool_choice != "none":
            parsed_calls = parse_tool_calls_from_response(generated_text)
            if parsed_calls:
                tool_calls = parsed_calls
                finish_reason = "tool_calls"
                # Remove tool call JSON from content if it exists
                # Handle both standalone JSON and JSON in code blocks
                content = generated_text
                
                # Remove JSON in code blocks
                content = re.sub(r'```(?:json)?\s*\n\{[^`]*"tool_calls"[^`]*\}\s*\n```', '', content, flags=re.DOTALL)
                
                # Remove standalone JSON with proper nesting
                # Find and remove complete JSON objects containing "tool_calls"
                if '"tool_calls"' in content:
                    # Try to find and remove the complete JSON structure
                    start_positions = [m.start() for m in re.finditer(r'\{', content)]
                    for start in reversed(start_positions):  # Process from end to avoid index issues
                        brace_count = 0
                        end = -1
                        for i, char in enumerate(content[start:], start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = i + 1
                                    break
                        
                        if end > start:
                            json_str = content[start:end]
                            if '"tool_calls"' in json_str:
                                try:
                                    # Verify it's valid JSON before removing
                                    json.loads(json_str)
                                    content = content[:start] + content[end:]
                                except:
                                    pass
                
                content = content.strip()

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
            # Create response message
            message = ChatMessage(role="assistant", content=content if content else None)
            if tool_calls:
                message.tool_calls = tool_calls

            # Create response
            response = ChatCompletionResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=message,
                        finish_reason=finish_reason,
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
    """Enhanced health check endpoint with graceful degradation status."""
    base_health = {
        "status": "healthy",
        "model_loaded": server_state["model"] is not None,
        "current_emotion": server_state["current_emotion"],
        "server_time": datetime.now().isoformat(),
    }
    
    # Add graceful degradation status
    try:
        # Import adaptive processor with new queue and vllm integration
        from adaptive_processor import get_adaptive_processor
        
        adaptive_processor = get_adaptive_processor()
        adaptive_stats = adaptive_processor.get_statistics() if adaptive_processor else {}
        
        # Get queue manager statistics
        queue_manager = get_request_queue_manager()
        queue_stats = queue_manager.get_statistics() if queue_manager else {}
        
        # Get vLLM wrapper statistics
        async_wrapper = get_async_vllm_wrapper()
        vllm_stats = async_wrapper.get_statistics() if async_wrapper else {}
        
        # Get health score from adaptive stats (fallback if health monitor not available)
        health_score = adaptive_stats.get("current_health_score", 1.0)
        
        # Determine overall status based on queue and wrapper state
        if queue_stats.get("status") == "critical" or vllm_stats.get("thread_capacity_used", 0) > 90:
            base_health["status"] = "critical"
        elif queue_stats.get("status") == "high" or vllm_stats.get("timeout_rate", 0) > 30:
            base_health["status"] = "degraded"
        elif queue_stats.get("capacity_percent", 0) > 50 or vllm_stats.get("timeout_rate", 0) > 10:
            base_health["status"] = "stressed"
        
        base_health.update({
            "graceful_degradation": {
                "health_score": health_score,
                "current_strategy": adaptive_stats.get("current_strategy", "healthy"),
                "optimization_rate": adaptive_stats.get("optimization_rate", 0),
                "rejection_rate": adaptive_stats.get("rejection_rate", 0),
                "queue_statistics": queue_stats,
                "vllm_statistics": vllm_stats,
                "adaptive_processor": {
                    "total_requests": adaptive_stats.get("total_requests", 0),
                    "optimized_requests": adaptive_stats.get("optimized_requests", 0),
                    "rejected_requests": adaptive_stats.get("rejected_requests", 0)
                }
            }
        })
        
    except Exception as e:
        logger.warning(f"Failed to get graceful degradation status: {e}")
        base_health["graceful_degradation"] = {"error": str(e)}
    
    return base_health


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

    # Performance optimization arguments
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Maximum batch size for request batching"
    )
    parser.add_argument(
        "--batch_timeout", type=float, default=0.05, help="Batch timeout in seconds"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.90, help="GPU memory utilization"
    )
    parser.add_argument("--max_num_seqs", type=int, default=64, help="Maximum number of sequences")
    parser.add_argument("--disable_batching", action="store_true", help="Disable request batching")
    parser.add_argument(
        "--enable_chunked_prefill", action="store_true", default=True, help="Enable chunked prefill"
    )
    
    # Phase 4.1: Graceful degradation configuration arguments
    parser.add_argument(
        "--request_timeout", type=int, default=60, 
        help="Request timeout in seconds for vLLM operations (default: 60)"
    )
    parser.add_argument(
        "--max_queue_size", type=int, default=50,
        help="Maximum number of requests in queue (default: 50)"
    )
    parser.add_argument(
        "--max_concurrent_requests", type=int, default=3,
        help="Maximum concurrent processing requests (default: 3)"
    )
    parser.add_argument(
        "--queue_rejection_threshold", type=float, default=0.8,
        help="Queue fullness threshold for rejection, 0.0-1.0 (default: 0.8)"
    )
    
    # Stage 2 configuration arguments
    parser.add_argument(
        "--reset_interval", type=int, default=300,
        help="Interval in seconds to reset abandoned thread counter (default: 300)"
    )
    parser.add_argument(
        "--vllm_rejection_threshold", type=float, default=0.7,
        help="Capacity threshold to start probabilistic rejection, 0.0-1.0 (default: 0.7)"
    )

    args = parser.parse_args()

    # Validate emotion
    valid_emotions = Emotions.get_emotions()
    if args.emotion not in valid_emotions:
        logger.error(f"Invalid emotion '{args.emotion}'. Valid emotions: {valid_emotions}")
        sys.exit(1)

    # Set server start time
    server_state["server_start_time"] = time.time()
    server_state["model_name"] = args.model_name

    # Initialize model with optimized settings
    try:
        initialize_model(
            model_path=args.model,
            emotion=args.emotion,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            enable_chunked_prefill=args.enable_chunked_prefill,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            disable_batching=args.disable_batching,
            request_timeout=args.request_timeout,
            max_queue_size=args.max_queue_size,
            max_concurrent_requests=args.max_concurrent_requests,
            queue_rejection_threshold=args.queue_rejection_threshold,
            reset_interval=args.reset_interval,
            vllm_rejection_threshold=args.vllm_rejection_threshold,
        )
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
