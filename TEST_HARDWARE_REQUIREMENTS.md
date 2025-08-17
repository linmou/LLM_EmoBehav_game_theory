# Test Hardware Requirements Guide

## 🖥️ **Test Categorization by Hardware Needs**

### **✅ CPU-Only Tests (No GPU Required)**

#### **Pure CPU Tests:**
```bash
# These tests run on any machine with Python/PyTorch
CUDA_VISIBLE_DEVICES="" python neuro_manipulation/repe/tests/test_multimodal_rep_reading_fixed.py
# Result: 8/8 tests pass (uses Mock objects only)

# Basic prompt format tests  
python test_prompt_format_integration.py
# Result: 4/4 tests pass (tokenizer loading only)
```

**Why CPU-Only:**
- Uses `Mock()` objects for models, tokenizers, processors
- Tests logic flow without actual model inference
- Only validates data structures and method calls

#### **Conditional CPU Tests:**
```bash
# Can work CPU-only if model files are cached locally
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/path/to/cached/qwen', local_files_only=True)
print('✅ Can run with cached tokenizer')
"
```

---

### **❌ GPU-Required Tests (Need ~6-8GB VRAM)**

#### **Real Model Tests:**
```bash
# These REQUIRE GPU with sufficient VRAM
CUDA_VISIBLE_DEVICES=3 python -m pytest neuro_manipulation/repe/tests/test_real_multimodal_integration.py::TestRealMultimodalIntegration::test_emotion_vector_extraction_basic -v -s

# What they do:
# - Load 3B parameter Qwen2.5-VL model
# - Perform forward passes with hidden state extraction
# - Use torch.bfloat16 (GPU optimization)
# - Require ~6-8GB VRAM
```

**Why GPU Required:**
- Loads actual Qwen2.5-VL-3B-Instruct model (3B parameters)
- Performs forward passes: `model(**inputs, output_hidden_states=True)`
- Uses `device_map="auto"` for GPU placement
- Memory requirement: ~6-8GB VRAM

---

## 🔄 **Making Tests More CPU-Friendly**

### **1. Environment Detection System**

I've created a comprehensive test environment system:

```bash
# Check your system capabilities
python neuro_manipulation/repe/tests/test_environment.py

# Example output:
# 🔍 Test Environment Analysis
# GPU: NVIDIA RTX 4090, VRAM: 24.0GB (need 8GB+)
# RAM: 32.0GB (need 16GB+ for CPU inference)  
# Model Files Available: True
# 🎯 Recommended Test Mode: full_gpu
```

### **2. CPU-Friendly Test Suite**

Created dedicated CPU-only tests that work without any GPU:

```bash
# Pure CPU tests (no model loading)
python neuro_manipulation/repe/tests/test_multimodal_cpu_friendly.py

# These tests verify:
# ✅ Input detection logic
# ✅ Prompt formatting
# ✅ Configuration validation
# ✅ Pattern matching algorithms
```

### **3. Smart Test Runner**

The smart test runner automatically detects your system and runs appropriate tests:

```bash
# Auto-detect and run suitable tests
python run_tests_by_capability.py

# Force specific modes
python run_tests_by_capability.py --mode cpu_only   # CPU-safe tests only
python run_tests_by_capability.py --mode mock_only  # Mock tests only
python run_tests_by_capability.py --mode gpu_only   # GPU tests only
python run_tests_by_capability.py --info           # Show system info
```

---

## 📊 **Complete Test Classification**

### **✅ Always Available (Mock-Based)**
```bash
# 8/8 tests - Pure Python logic
python neuro_manipulation/repe/tests/test_multimodal_rep_reading_fixed.py
# Time: ~0.1 seconds
# Memory: <100MB RAM
# Requirements: Python + PyTorch (CPU)
```

**What these test:**
- Multimodal input detection logic
- Prompt format integration with mocks
- Model layer detection patterns  
- QwenVL format functionality
- Tokenizer validation logic
- Manual format selection
- Batch input detection
- Configuration structure validation

### **🖥️ CPU-Friendly (No GPU)**
```bash
# New CPU-friendly test suite
python neuro_manipulation/repe/tests/test_multimodal_cpu_friendly.py
# Time: ~1-5 seconds
# Memory: ~500MB RAM
# Requirements: Python + PyTorch + optional cached tokenizer
```

**What these test:**
- Environment detection system
- CPU-based input processing
- Prompt formatting without real models
- Pattern-based model detection
- Tokenizer integration (if cached)
- Configuration file validation
- Batch processing logic

### **⚠️ Tokenizer Required (Light GPU/CPU)**
```bash
# Requires model files or internet connection
python test_prompt_format_integration.py
# Time: ~5-10 seconds  
# Memory: ~1-2GB RAM
# Requirements: Tokenizer loading capability
```

**What these test:**
- Real tokenizer integration
- Token ID validation
- Format detection with actual models
- Vision token verification

### **❌ GPU Required (Heavy)**
```bash
# Requires 6-8GB VRAM + model files
CUDA_VISIBLE_DEVICES=3 python -m pytest neuro_manipulation/repe/tests/test_real_multimodal_integration.py::TestRealMultimodalIntegration::test_emotion_vector_extraction_basic -v -s
# Time: ~10-15 seconds
# Memory: ~6-8GB VRAM + 4GB RAM
# Requirements: GPU + full model files
```

**What these test:**
- Real model loading and inference
- Forward pass with hidden state extraction
- Complete emotion vector extraction pipeline
- Token/feature alignment verification
- End-to-end multimodal processing

---

## 🎯 **Quick Test Commands by System Type**

### **💻 Laptop/CPU-Only Systems:**
```bash
# Just run the essentials
python run_tests_by_capability.py --mode cpu_only

# Or manually:
python neuro_manipulation/repe/tests/test_multimodal_rep_reading_fixed.py
python neuro_manipulation/repe/tests/test_multimodal_cpu_friendly.py
```

### **🖥️ Workstation with GPU:**
```bash
# Run everything
python run_tests_by_capability.py --mode all

# Or auto-detect
python run_tests_by_capability.py
```

### **☁️ Server/Cluster:**
```bash
# Check capabilities first
python run_tests_by_capability.py --info

# Then run appropriate level
export TEST_MODE=gpu_required
python run_tests_by_capability.py
```

### **🐳 Docker/CI Systems:**
```bash
# Mock-only for CI
export TEST_MODE=mock_only  
python run_tests_by_capability.py --mode mock_only
```

---

## 🔧 **Making GPU Tests CPU-Compatible**

### **CPU-Safe Model Loading:**
```python
# Instead of:
model = AutoModel.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16)

# Use adaptive loading:
from neuro_manipulation.repe.tests.test_environment import TestEnvironment
device_config = TestEnvironment.get_device_config()
model = AutoModel.from_pretrained(path, **device_config)

# Results in:
# GPU system: device_map="auto", torch_dtype=torch.bfloat16  
# CPU system: device_map="cpu", torch_dtype=torch.float32
```

### **Skip Decorators:**
```python
from neuro_manipulation.repe.tests.test_environment import require_gpu, require_model_files

@require_gpu
def test_gpu_inference():
    # This test automatically skips on systems without sufficient GPU
    pass

@require_model_files  
def test_with_real_model():
    # This test skips if model files aren't available
    pass
```

---

## 📈 **Test Coverage by System Type**

| System Type | Mock Tests | CPU Tests | Tokenizer Tests | GPU Tests | Total Coverage |
|-------------|------------|-----------|-----------------|-----------|----------------|
| **Laptop (CPU-only)** | ✅ 8/8 | ✅ 6/6 | ⚠️ 0-4/4* | ❌ 0/2 | ~70-85% |
| **Workstation (GPU)** | ✅ 8/8 | ✅ 6/6 | ✅ 4/4 | ✅ 2/2 | 100% |
| **Server (No Models)** | ✅ 8/8 | ✅ 4/6 | ❌ 0/4 | ❌ 0/2 | ~60% |
| **CI/Docker** | ✅ 8/8 | ✅ 4/6 | ❌ 0/4 | ❌ 0/2 | ~60% |

*Depends on cached tokenizers or internet access

---

## ✅ **Benefits of This Approach**

### **For Developers:**
- ✅ **Universal Testing**: Core logic testable on any system
- ✅ **Fast Iteration**: Mock tests run in <1 second  
- ✅ **Clear Requirements**: Know exactly what each test needs
- ✅ **Graceful Degradation**: Tests skip with clear messages

### **For CI/CD:**
- ✅ **Predictable**: Always know which tests will run
- ✅ **Scalable**: Different test levels for different environments
- ✅ **Cost-Effective**: Don't need GPU instances for all tests

### **For Research:**
- ✅ **Accessible**: Anyone can contribute and test logic changes
- ✅ **Comprehensive**: Full validation when resources are available
- ✅ **Reliable**: Consistent behavior across different systems

The system now automatically adapts to whatever hardware is available while maintaining comprehensive test coverage! 🎯