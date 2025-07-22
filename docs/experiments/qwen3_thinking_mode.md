# Qwen3 Thinking Mode Implementation

## Overview

Qwen3 thinking mode is a feature that enables models to generate structured internal reasoning before providing their final answer. This documentation covers the implementation, configuration, and usage of thinking mode in the neuro-manipulation framework.

## How Thinking Mode Works

### 1. Prompt Format Enhancement
- The system automatically detects Qwen3 models using pattern matching
- When `enable_thinking: true` is set, the `/think` tag is injected into user messages
- The model generates reasoning in `<think>...</think>` blocks before the final answer

### 2. Token Requirements
- **Standard experiments**: 440 tokens (sufficient for most responses)
- **Thinking mode**: 1800 tokens recommended (includes reasoning + answer)
- **Minimum required**: ~1100-1300 tokens based on analysis

## Configuration

### Basic Setup
```yaml
llm:
  generation_config:
    max_new_tokens: 1800  # Increased for thinking mode
    enable_thinking: true # Activates thinking mode
    temperature: 0.7
    top_p: 0.95
```

### Model Detection
Thinking mode is automatically activated for models matching:
- `Qwen3` (case-insensitive)
- Examples: `Qwen3-1.7B`, `Qwen/Qwen3-4B`, `/path/to/Qwen3-8B`

## Files and Configurations

### Production Configurations
- `config/Qwen3_Thinking_Mode_Enabled.yaml` - Full experiment with thinking mode
- `config/Qwen3_Thinking_Mode_Comparison.yaml` - Baseline without thinking mode

### Test Configurations
- `config/Qwen3_Thinking_Mode_Test_Enabled.yaml` - Quick test (5 scenarios)
- `config/Qwen3_Thinking_Mode_Test_Baseline.yaml` - Quick baseline test
- `config/Qwen3_Thinking_Mode_Minimal_Test.yaml` - Minimal test (2 scenarios)

### Analysis Tools
- `test_qwen3_thinking_length.py` - Token usage analysis
- `run_qwen3_thinking_mode_test.py` - Quick validation runner
- `run_qwen3_thinking_mode_experiment.py` - Full experiment runner

## Usage Examples

### Running Full Experiments
```bash
# Full comparison experiment (baseline + thinking mode)
python run_qwen3_thinking_mode_experiment.py

# Individual experiments
python -m neuro_manipulation.experiment_series_runner --config config/Qwen3_Thinking_Mode_Enabled.yaml
```

### Quick Testing
```bash
# Test token requirements
python test_qwen3_thinking_length.py

# Quick validation (5 scenarios)
python run_qwen3_thinking_mode_test.py

# Minimal test (2 scenarios)
python run_minimal_thinking_test.py
```

## Implementation Details

### Code Components

#### 1. Prompt Format (`neuro_manipulation/prompt_formats.py`)
```python
class Qwen3InstFormat(PromptFormat):
    @staticmethod
    def name_pattern(model_name: str) -> bool:
        return "qwen3" in model_name.lower()
    
    @staticmethod
    def build(system_prompt: str, user_messages: List[str], enable_thinking: bool = False) -> str:
        # Injects /think tag when thinking mode is enabled
```

#### 2. Experiment Integration (`neuro_manipulation/experiments/emotion_game_experiment.py`)
- Passes `enable_thinking` parameter from config to prompt builder
- Handles longer token sequences automatically

#### 3. Prompt Wrapper (`neuro_manipulation/prompt_wrapper.py`)
- Routes thinking mode parameter to appropriate format class
- Maintains backward compatibility

## Performance Analysis

### Token Usage Patterns
Based on analysis of 12 test scenarios across different token limits:

| Token Limit | Avg Total | Avg Thinking | Truncated |
|-------------|-----------|--------------|-----------|
| 500         | 500       | 0            | 3/3       |
| 1000        | 872       | 755          | 1/3       |
| 1500        | 964       | 394          | 1/3       |
| 2000        | 1263      | 497          | 1/3       |

### Experiment Duration
- **Qwen3-1.7B**: ~32 minutes (baseline and thinking mode similar)
- **Qwen3-4B**: ~41-46 minutes
- **Qwen3-8B**: ~57 minutes
- Main bottleneck: Model loading/inference, not token generation

## Best Practices

### 1. Token Configuration
- Use **1800+ tokens** for thinking mode experiments
- Monitor for truncation in results
- Use test configurations for validation

### 2. Experiment Design
- Start with test configurations (5 scenarios) for validation
- Use minimal test (2 scenarios) for quick checks
- Run full experiments (36 scenarios) only after validation

### 3. Performance Optimization
- Use smaller models (1.7B) for initial testing
- Consider batch size adjustments for throughput
- Monitor GPU memory usage with larger models

## Troubleshooting

### Common Issues

#### 1. Truncated Outputs
**Symptom**: Incomplete responses or missing `</think>` tags
**Solution**: Increase `max_new_tokens` to 1800+

#### 2. No Thinking Tags
**Symptom**: Model doesn't generate `<think>` blocks
**Solution**: Verify `enable_thinking: true` and model name detection

#### 3. Long Experiment Times
**Symptom**: Experiments take hours to complete
**Solution**: Use test configurations first, then scale up

### Validation Steps
1. Run `python test_qwen3_thinking_length.py` to verify token requirements
2. Run `python run_minimal_thinking_test.py` for quick validation
3. Check results for proper thinking structure and no truncation

## Results Location

- **Full Results**: `results/Qwen3_Thinking_Mode_Comparison_[Baseline|Enabled]/`
- **Test Results**: `results/Qwen3_Thinking_Mode_Test_[Baseline|Enabled]/`
- **Token Analysis**: `qwen3_thinking_length_analysis_[timestamp].json`
- **Reports**: `reports/qwen3_thinking_mode_comparison_report_[timestamp].md`