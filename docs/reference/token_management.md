# Token Management Guide

## Overview

Token management is crucial for optimal experiment performance, especially when using advanced features like thinking mode or processing large datasets. This guide covers token configuration, analysis, and optimization strategies.

## Token Limits by Model Type

### Standard Models
- **Default**: 440 tokens (sufficient for most direct responses)
- **Recommended**: 500-600 tokens for complex scenarios
- **Maximum**: Usually 1000 tokens for very detailed responses

### Thinking Mode Models (Qwen3)
- **Minimum**: 1200 tokens (includes reasoning + answer)
- **Recommended**: 1800 tokens (safe buffer for complex reasoning)
- **Observed Range**: 400-1300 tokens based on scenario complexity

### Large Context Models
- **Sequential Games**: 1200+ tokens (for game history)
- **Multi-turn Scenarios**: 1500+ tokens (for conversation context)

## Configuration Guidelines

### Basic Token Configuration
```yaml
llm:
  generation_config:
    max_new_tokens: 440  # Adjust based on model type
    temperature: 0.7
    top_p: 0.95
```

### Thinking Mode Configuration
```yaml
llm:
  generation_config:
    max_new_tokens: 1800  # Increased for internal reasoning
    enable_thinking: true
    temperature: 0.7
    top_p: 0.95
```

### Game-Specific Adjustments
```yaml
# For sequential games with history
max_new_tokens: 1200

# For simple simultaneous games
max_new_tokens: 440

# For complex multi-option scenarios
max_new_tokens: 600
```

## Token Analysis Tools

### 1. Built-in Analysis
Most experiments automatically log token usage in results:
```json
{
  "token_stats": {
    "avg_tokens": 285,
    "max_tokens": 440,
    "truncated_responses": 0
  }
}
```

### 2. Qwen3 Thinking Mode Analyzer
```bash
python test_qwen3_thinking_length.py
```

Output includes:
- Token distribution across different limits
- Thinking vs answer token breakdown
- Truncation analysis
- Recommended token limits

### 3. Custom Token Analysis
For analyzing existing results:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
token_count = len(tokenizer.encode(response_text))
```

## Performance Implications

### Token Limit vs Speed
- **Lower limits**: Faster generation, risk of truncation
- **Higher limits**: Slower generation, complete responses
- **Optimal range**: 1.2-1.5x expected token usage

### Memory Considerations
- GPU memory scales with: `batch_size × max_tokens × model_size`
- Reduce batch size if increasing token limits
- Monitor GPU utilization during experiments

### Cost Implications (API Models)
- API costs scale linearly with token usage
- Monitor total token consumption in experiments
- Use token limits to control costs

## Optimization Strategies

### 1. Progressive Testing
```bash
# Start with minimal test
python run_minimal_thinking_test.py  # 2 scenarios

# Scale to quick test
python run_qwen3_thinking_mode_test.py  # 5 scenarios

# Run full experiment
python run_qwen3_thinking_mode_experiment.py  # 36 scenarios
```

### 2. Dynamic Token Adjustment
```yaml
# Different limits for different emotions/scenarios
emotions:
  anger:
    max_new_tokens: 1800  # May generate longer reasoning
  happiness:
    max_new_tokens: 1200  # Usually more concise
```

### 3. Batch Size Optimization
```yaml
data:
  batch_size: 4  # Standard
  batch_size: 2  # For high token limits
  batch_size: 1  # For very long sequences
```

## Troubleshooting

### Truncated Responses
**Symptoms**:
- Incomplete sentences
- Missing closing tags (`</think>`)
- Cut-off JSON responses

**Solutions**:
1. Increase `max_new_tokens` by 50-100%
2. Analyze token usage patterns
3. Adjust based on specific model behavior

### Out of Memory Errors
**Symptoms**:
- CUDA OOM errors
- Process crashes during generation

**Solutions**:
1. Reduce `batch_size`
2. Lower `max_new_tokens` if possible
3. Use tensor parallelism for large models

### Slow Generation
**Symptoms**:
- Long wait times per response
- Low tokens/second throughput

**Solutions**:
1. Optimize token limits (not too high)
2. Adjust sampling parameters
3. Use faster hardware or parallel processing

## Best Practices

### 1. Model-Specific Configuration
- Test token requirements for each new model
- Document optimal settings in model configs
- Use different configs for different model families

### 2. Scenario-Aware Limits
- Simple games: 400-600 tokens
- Complex scenarios: 800-1200 tokens
- Thinking mode: 1500-2000 tokens
- Sequential games: 1000-1500 tokens

### 3. Monitoring and Logging
- Always log token usage statistics
- Monitor for truncation in results
- Track performance metrics vs token limits

### 4. Cost Management
- Use test configurations for development
- Optimize token limits before full runs
- Monitor total token consumption

## Example Configurations

### Research Experiment
```yaml
# Balanced for quality and efficiency
llm:
  generation_config:
    max_new_tokens: 600
    temperature: 0.7
    do_sample: true
```

### Production Deployment
```yaml
# Conservative limits for reliability
llm:
  generation_config:
    max_new_tokens: 440
    temperature: 0.1
    do_sample: false
```

### Thinking Mode Research
```yaml
# High limits for complete reasoning
llm:
  generation_config:
    max_new_tokens: 2000
    enable_thinking: true
    temperature: 0.7
```