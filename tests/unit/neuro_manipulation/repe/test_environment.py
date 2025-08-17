#!/usr/bin/env python3
"""
Test Environment Detection for Multimodal RepE Tests

This module provides utilities to detect available hardware and model resources,
allowing tests to run appropriately on different systems.
"""

import os
import torch
from pathlib import Path
import psutil
from typing import Dict, Optional
import sys

# Add project root to path for imports
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

class TestEnvironment:
    """Detect and manage test environment capabilities."""
    
    # Model requirements
    QWEN_VL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
    MIN_GPU_MEMORY_GB = 8  # Minimum VRAM for 3B model
    MIN_RAM_GB = 16       # Minimum RAM for CPU inference
    
    @classmethod
    def has_gpu_support(cls) -> bool:
        """Check if GPU is available with sufficient VRAM."""
        if not torch.cuda.is_available():
            return False
        
        try:
            # Check available VRAM
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)
            return vram_gb >= cls.MIN_GPU_MEMORY_GB
        except Exception:
            return False
    
    @classmethod  
    def has_sufficient_ram(cls) -> bool:
        """Check if system has sufficient RAM for CPU inference."""
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)
            return ram_gb >= cls.MIN_RAM_GB
        except Exception:
            return False
    
    @classmethod
    def has_model_files(cls) -> bool:
        """Check if required model files are available locally."""
        return Path(cls.QWEN_VL_MODEL_PATH).exists()
    
    @classmethod
    def can_load_tokenizer(cls) -> bool:
        """Check if we can load tokenizers (online or cached)."""
        try:
            from transformers import AutoTokenizer
            # Try to load from cache first
            tokenizer = AutoTokenizer.from_pretrained(
                cls.QWEN_VL_MODEL_PATH, 
                local_files_only=True,
                trust_remote_code=True
            )
            return True
        except Exception:
            # Try online if local fails
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    trust_remote_code=True
                )
                return True
            except Exception:
                return False
    
    @classmethod
    def get_test_mode(cls) -> str:
        """Determine optimal test mode based on available resources."""
        # Check environment override
        env_mode = os.getenv("TEST_MODE", "").lower()
        if env_mode in ["cpu_only", "mock_only", "gpu_required"]:
            return env_mode
        
        # Auto-detect based on capabilities
        if cls.has_gpu_support() and cls.has_model_files():
            return "full_gpu"
        elif cls.has_sufficient_ram() and cls.has_model_files():
            return "cpu_inference"
        elif cls.can_load_tokenizer():
            return "tokenizer_only"
        else:
            return "mock_only"
    
    @classmethod
    def get_device_config(cls) -> Dict[str, any]:
        """Get device configuration for model loading."""
        mode = cls.get_test_mode()
        
        if mode == "full_gpu":
            return {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True
            }
        elif mode == "cpu_inference":
            return {
                "device_map": "cpu", 
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True
            }
        else:
            return {"device_map": "cpu", "torch_dtype": torch.float32}
    
    @classmethod
    def get_skip_reason(cls, test_requirement: str) -> Optional[str]:
        """Get skip reason for tests based on requirements."""
        mode = cls.get_test_mode()
        
        skip_conditions = {
            "gpu_required": {
                "condition": not cls.has_gpu_support(),
                "reason": f"Requires GPU with {cls.MIN_GPU_MEMORY_GB}GB+ VRAM"
            },
            "model_required": {
                "condition": not cls.has_model_files(),
                "reason": f"Model files not found at {cls.QWEN_VL_MODEL_PATH}"
            },
            "tokenizer_required": {
                "condition": not cls.can_load_tokenizer(),
                "reason": "Cannot load tokenizer (offline and no cache)"
            },
            "inference_required": {
                "condition": mode == "mock_only",
                "reason": "No inference capability available"
            }
        }
        
        condition_info = skip_conditions.get(test_requirement, {})
        if condition_info.get("condition", False):
            return condition_info.get("reason", f"Cannot meet requirement: {test_requirement}")
        
        return None
    
    @classmethod
    def print_environment_info(cls):
        """Print detailed environment information."""
        print("🔍 Test Environment Analysis")
        print("=" * 50)
        
        # GPU Info
        gpu_available = torch.cuda.is_available()
        print(f"CUDA Available: {gpu_available}")
        if gpu_available:
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                vram_gb = gpu_props.total_memory / (1024**3)
                print(f"GPU: {gpu_props.name}")
                print(f"VRAM: {vram_gb:.1f}GB (need {cls.MIN_GPU_MEMORY_GB}GB+)")
                print(f"GPU Suitable: {cls.has_gpu_support()}")
            except Exception as e:
                print(f"GPU Info Error: {e}")
        
        # RAM Info
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)
            print(f"RAM: {ram_gb:.1f}GB (need {cls.MIN_RAM_GB}GB+ for CPU inference)")
            print(f"RAM Sufficient: {cls.has_sufficient_ram()}")
        except Exception as e:
            print(f"RAM Info Error: {e}")
        
        # Model Files
        print(f"Model Files Available: {cls.has_model_files()}")
        print(f"Tokenizer Loadable: {cls.can_load_tokenizer()}")
        
        # Recommended Mode
        mode = cls.get_test_mode()
        print(f"\n🎯 Recommended Test Mode: {mode}")
        
        # Test Capabilities
        print("\n📋 Test Capabilities:")
        tests = [
            ("Mock-based tests", "Always available"),
            ("Tokenizer tests", "✅" if cls.can_load_tokenizer() else "❌"),
            ("CPU inference tests", "✅" if cls.has_sufficient_ram() and cls.has_model_files() else "❌"),
            ("GPU inference tests", "✅" if cls.has_gpu_support() and cls.has_model_files() else "❌"),
        ]
        
        for test_name, status in tests:
            print(f"  {test_name:20} {status}")


def require_gpu(func):
    """Decorator to skip tests that require GPU."""
    import pytest
    skip_reason = TestEnvironment.get_skip_reason("gpu_required")
    if skip_reason:
        return pytest.mark.skip(reason=skip_reason)(func)
    return func


def require_model_files(func):
    """Decorator to skip tests that require model files."""
    import pytest
    skip_reason = TestEnvironment.get_skip_reason("model_required")
    if skip_reason:
        return pytest.mark.skip(reason=skip_reason)(func)
    return func


def require_tokenizer(func):
    """Decorator to skip tests that require tokenizer."""
    import pytest
    skip_reason = TestEnvironment.get_skip_reason("tokenizer_required")
    if skip_reason:
        return pytest.mark.skip(reason=skip_reason)(func)
    return func


def require_inference(func):
    """Decorator to skip tests that require inference capability."""
    import pytest
    skip_reason = TestEnvironment.get_skip_reason("inference_required")
    if skip_reason:
        return pytest.mark.skip(reason=skip_reason)(func)
    return func


if __name__ == "__main__":
    # Print environment info when run directly
    TestEnvironment.print_environment_info()
    
    print("\n" + "=" * 50)
    print("Environment Variable Options:")
    print("export TEST_MODE=cpu_only     # Force CPU-only tests")
    print("export TEST_MODE=mock_only    # Force mock-based tests only") 
    print("export TEST_MODE=gpu_required # Force GPU tests (will fail if no GPU)")
    print("# (leave unset for auto-detection)")