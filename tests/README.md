# Tests Directory Structure

This directory contains all tests for the LLM_EmoBehav_game_theory_multimodal project, organized by test type and hardware requirements.

## Directory Structure

```
tests/
├── README.md                          # This file - test organization guide
├── run_tests_by_capability.py         # Smart test runner that adapts to hardware
├── unit/                              # Unit tests - fast, isolated, minimal dependencies
│   ├── neuro_manipulation/
│   │   └── repe/
│   │       ├── test_environment.py    # Hardware detection and test environment utilities
│   │       └── test_multimodal_rep_reading.py  # Mock-based multimodal RepE tests
│   └── test_prompt_format_integration.py       # Prompt format system tests
├── integration/                       # Integration tests - require real models/GPU
│   └── test_real_multimodal_integration.py     # End-to-end multimodal emotion extraction
└── cpu_friendly/                      # CPU-only tests - no GPU required
    └── test_multimodal_cpu_friendly.py         # Lightweight tests for any system
```

## Test Categories

### 🚀 **Mock-Based Unit Tests** (Always Available)
- **Location**: `tests/unit/`
- **Requirements**: Python + PyTorch (CPU)
- **Runtime**: < 1 second
- **What they test**: Core logic, input detection, prompt formatting
- **Files**: 
  - `test_multimodal_rep_reading.py` - Mock-based multimodal RepE tests
  - `test_prompt_format_integration.py` - Prompt format system validation

### 🖥️ **CPU-Friendly Tests** (No GPU Required)
- **Location**: `tests/cpu_friendly/`
- **Requirements**: Python + PyTorch + optional cached tokenizers
- **Runtime**: 1-5 seconds
- **What they test**: Environment detection, pattern matching, configuration validation
- **Files**:
  - `test_multimodal_cpu_friendly.py` - Comprehensive CPU-only test suite

### ⚡ **Integration Tests** (GPU Required)
- **Location**: `tests/integration/`
- **Requirements**: GPU with 6-8GB VRAM + model files
- **Runtime**: 10-15 seconds
- **What they test**: Real model inference, emotion vector extraction, end-to-end pipelines
- **Files**:
  - `test_real_multimodal_integration.py` - Full multimodal emotion extraction tests

## Running Tests

### Quick Commands

```bash
# Auto-detect hardware and run appropriate tests
python tests/run_tests_by_capability.py

# Force specific test levels
python tests/run_tests_by_capability.py --mode cpu_only    # CPU-safe tests only
python tests/run_tests_by_capability.py --mode mock_only   # Mock tests only
python tests/run_tests_by_capability.py --mode gpu_only    # GPU tests only
python tests/run_tests_by_capability.py --mode all         # All tests (if hardware supports)

# Show system capabilities
python tests/run_tests_by_capability.py --info
```

### Individual Test Files

```bash
# Unit tests (always work)
python tests/unit/neuro_manipulation/repe/test_multimodal_rep_reading.py
python tests/unit/test_prompt_format_integration.py

# CPU-friendly tests
python tests/cpu_friendly/test_multimodal_cpu_friendly.py

# Integration tests (GPU required)
CUDA_VISIBLE_DEVICES=3 python -m pytest tests/integration/test_real_multimodal_integration.py -v -s
```

## Hardware Requirements

| Test Type | RAM | VRAM | Model Files | Internet | Runtime |
|-----------|-----|------|-------------|----------|---------|
| **Mock Tests** | Any | None | No | No | <1s |
| **CPU Tests** | 4GB+ | None | Optional | Optional | 1-5s |
| **Integration** | 8GB+ | 6-8GB | Yes | No | 10-15s |

## Test Environment Detection

The `test_environment.py` module automatically detects:
- ✅ **GPU availability and VRAM**
- ✅ **System RAM**
- ✅ **Model file availability**
- ✅ **Tokenizer loading capability**
- ✅ **Recommended test mode**

Tests use decorators to skip gracefully when requirements aren't met:
```python
@require_gpu
def test_gpu_inference():
    # Automatically skips on systems without sufficient GPU
    pass

@require_model_files  
def test_with_real_model():
    # Skips if model files aren't available
    pass
```

## Test Coverage by System Type

| System Type | Mock Tests | CPU Tests | Integration | Total Coverage |
|-------------|------------|-----------|-------------|----------------|
| **Laptop (CPU-only)** | ✅ 100% | ✅ 100% | ❌ 0% | ~70% |
| **Workstation (GPU)** | ✅ 100% | ✅ 100% | ✅ 100% | 100% |
| **Server (No Models)** | ✅ 100% | ⚠️ Partial | ❌ 0% | ~60% |
| **CI/Docker** | ✅ 100% | ⚠️ Partial | ❌ 0% | ~60% |

## Benefits of This Structure

### For Developers:
- ✅ **Universal Testing**: Core logic testable on any system
- ✅ **Fast Iteration**: Mock tests run in <1 second  
- ✅ **Clear Requirements**: Know exactly what each test needs
- ✅ **Graceful Degradation**: Tests skip with clear messages

### For CI/CD:
- ✅ **Predictable**: Always know which tests will run
- ✅ **Scalable**: Different test levels for different environments
- ✅ **Cost-Effective**: Don't need GPU instances for all tests

### For Research:
- ✅ **Accessible**: Anyone can contribute and test logic changes
- ✅ **Comprehensive**: Full validation when resources are available
- ✅ **Reliable**: Consistent behavior across different systems

## Related Documentation

- `TEST_HARDWARE_REQUIREMENTS.md` - Detailed hardware analysis
- `docs/MULTIMODAL_REPE.md` - Multimodal RepE implementation guide
- `CLAUDE.md` - Development environment setup