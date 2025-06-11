# Sequence Probability vLLM Hook Fix

## Issue Summary

The `SequenceProbVLLMHook` was returning empty results instead of calculating sequence probabilities. The test `test_sequence_prob_tp_consistency.py` was failing with:

```
AssertionError: 0 != 1
```

This indicated that `get_log_prob()` was returning an empty list instead of the expected single result.

## Root Cause Analysis

The investigation revealed multiple issues:

### 1. **Undefined Variable Bug**
- **Problem**: The hook function `hook_fn_sequence_prob` referenced an undefined variable `rank`
- **Impact**: This would cause a `NameError` when the hook was called
- **Fix**: Added proper rank detection using `dist.get_rank()` if distributed, otherwise default to 0

### 2. **Hook Never Called**
- **Problem**: The forward hooks registered on the LM head were never being triggered during vLLM generation
- **Root Cause**: vLLM v1 engine uses optimizations that bypass normal PyTorch module execution paths
- **Evidence**: No "TEST HOOK CALLED" messages appeared during generation, confirming hooks weren't executed

### 3. **Wrong Approach for vLLM v1**
- **Problem**: The original approach tried to intercept logits via forward hooks, which doesn't work with vLLM v1's optimized execution
- **Solution**: Switch to using vLLM's built-in logprobs output from generation

### 4. **Token Encoding Mismatch**
- **Problem**: Target sequence "Paris" encoded as token 40313, but vLLM logprobs contained token 6342 ("ĠParis" with space)
- **Impact**: Target tokens weren't found in the logprobs, resulting in zero probability
- **Fix**: Added fallback to check both versions (with and without leading space)

### 5. **Logprobs Limit**
- **Problem**: Initial code tried to request 50,000 logprobs, but vLLM v1 has a maximum limit of 20
- **Fix**: Reduced logprobs parameter to 20 (the maximum allowed)

## Solution Implemented

### New Architecture

1. **Removed Hook-Based Approach**: Eliminated the attempt to capture logits via forward hooks
2. **Direct Logprobs Extraction**: Use vLLM's native logprobs output from generation
3. **Dual Token Encoding**: Check both regular and space-prefixed versions of target tokens
4. **Proper Parameter Limits**: Respect vLLM v1's logprobs limit of 20

### Key Changes

1. **New Method**: `_calculate_sequence_probabilities_from_output()` 
   - Extracts probabilities directly from vLLM generation output
   - Handles both token encoding variants

2. **Updated Token Handling**:
   ```python
   # Original: only check target tokens
   tokens = self.tokenizer.encode(seq, add_special_tokens=False)
   
   # Fixed: check both versions
   tokens = self.tokenizer.encode(seq, add_special_tokens=False)
   tokens_with_space = self.tokenizer.encode(' ' + seq, add_special_tokens=False)
   ```

3. **Corrected Parameters**:
   ```python
   # Fixed logprobs limit
   logprobs=20  # Max allowed in vLLM v1 is 20
   ```

## Results

### Before Fix
- **TP=1**: Empty results `[]`
- **TP=2**: Empty results `[]`
- **Test**: Failed with `AssertionError: 0 != 1`

### After Fix  
- **TP=1**: `log_prob=-3.4128, prob=0.0329, perplexity=30.35`
- **TP=2**: `log_prob=-3.4494, prob=0.0318, perplexity=31.48`
- **Test**: Now properly tests tensor parallel consistency (reveals ~0.037 difference)

## Outstanding Issue

The test now reveals a legitimate **tensor parallel consistency issue**: there's a small numerical difference (~0.037 log probability units) between single-GPU and tensor-parallel results. This suggests potential precision differences in the tensor parallel implementation and warrants further investigation.

## Technical Notes

- **vLLM Version**: v0.8.3 with V1 engine
- **Architecture**: The fix is compatible with both single-GPU and tensor-parallel configurations
- **Performance**: Using native vLLM logprobs is more efficient than hook-based interception
- **Maintainability**: Cleaner approach that relies on vLLM's documented API rather than internal hooks 

## Issue Summary

The `SequenceProbVLLMHook` was returning empty results instead of calculating sequence probabilities. The test `test_sequence_prob_tp_consistency.py` was failing with:

```
AssertionError: 0 != 1
```

This indicated that `get_log_prob()` was returning an empty list instead of the expected single result.

## Root Cause Analysis

The investigation revealed multiple issues:

### 1. **Undefined Variable Bug**
- **Problem**: The hook function `hook_fn_sequence_prob` referenced an undefined variable `rank`
- **Impact**: This would cause a `NameError` when the hook was called
- **Fix**: Added proper rank detection using `dist.get_rank()` if distributed, otherwise default to 0

### 2. **Hook Never Called**
- **Problem**: The forward hooks registered on the LM head were never being triggered during vLLM generation
- **Root Cause**: vLLM v1 engine uses optimizations that bypass normal PyTorch module execution paths
- **Evidence**: No "TEST HOOK CALLED" messages appeared during generation, confirming hooks weren't executed

### 3. **Wrong Approach for vLLM v1**
- **Problem**: The original approach tried to intercept logits via forward hooks, which doesn't work with vLLM v1's optimized execution
- **Solution**: Switch to using vLLM's built-in logprobs output from generation

### 4. **Token Encoding Mismatch**
- **Problem**: Target sequence "Paris" encoded as token 40313, but vLLM logprobs contained token 6342 ("ĠParis" with space)
- **Impact**: Target tokens weren't found in the logprobs, resulting in zero probability
- **Fix**: Added fallback to check both versions (with and without leading space)

### 5. **Logprobs Limit**
- **Problem**: Initial code tried to request 50,000 logprobs, but vLLM v1 has a maximum limit of 20
- **Fix**: Reduced logprobs parameter to 20 (the maximum allowed)

## Solution Implemented

### New Architecture

1. **Removed Hook-Based Approach**: Eliminated the attempt to capture logits via forward hooks
2. **Direct Logprobs Extraction**: Use vLLM's native logprobs output from generation
3. **Dual Token Encoding**: Check both regular and space-prefixed versions of target tokens
4. **Proper Parameter Limits**: Respect vLLM v1's logprobs limit of 20

### Key Changes

1. **New Method**: `_calculate_sequence_probabilities_from_output()` 
   - Extracts probabilities directly from vLLM generation output
   - Handles both token encoding variants

2. **Updated Token Handling**:
   ```python
   # Original: only check target tokens
   tokens = self.tokenizer.encode(seq, add_special_tokens=False)
   
   # Fixed: check both versions
   tokens = self.tokenizer.encode(seq, add_special_tokens=False)
   tokens_with_space = self.tokenizer.encode(' ' + seq, add_special_tokens=False)
   ```

3. **Corrected Parameters**:
   ```python
   # Fixed logprobs limit
   logprobs=20  # Max allowed in vLLM v1 is 20
   ```

## Results

### Before Fix
- **TP=1**: Empty results `[]`
- **TP=2**: Empty results `[]`
- **Test**: Failed with `AssertionError: 0 != 1`

### After Fix  
- **TP=1**: `log_prob=-3.4128, prob=0.0329, perplexity=30.35`
- **TP=2**: `log_prob=-3.4494, prob=0.0318, perplexity=31.48`
- **Test**: Now properly tests tensor parallel consistency (reveals ~0.037 difference)

## Outstanding Issue

The test now reveals a legitimate **tensor parallel consistency issue**: there's a small numerical difference (~0.037 log probability units) between single-GPU and tensor-parallel results. This suggests potential precision differences in the tensor parallel implementation and warrants further investigation.

## Technical Notes

- **vLLM Version**: v0.8.3 with V1 engine
- **Architecture**: The fix is compatible with both single-GPU and tensor-parallel configurations
- **Performance**: Using native vLLM logprobs is more efficient than hook-based interception
- **Maintainability**: Cleaner approach that relies on vLLM's documented API rather than internal hooks 