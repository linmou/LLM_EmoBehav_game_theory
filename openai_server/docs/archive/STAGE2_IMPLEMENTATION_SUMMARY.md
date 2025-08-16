# Stage 2 Implementation Summary

## What Was Implemented

Modified `async_vllm_wrapper.py` to add Stage 2 graceful degradation features:

### 1. Periodic Reset (Stage 1 feature)
```python
# Reset counter every 5 minutes
if current_time - self.last_reset_time > self.reset_interval:
    self.abandoned_threads = 0
    self.last_reset_time = current_time
```

### 2. Graduated Response (Stage 2 feature)
```python
# Probabilistic rejection between 70-100% capacity
capacity_used = self.abandoned_threads / self.max_abandoned_threads

if capacity_used >= 1.0:
    # Reject all at 100%
    raise HTTPException(503, "Server at full capacity")
elif capacity_used >= 0.7:
    # Probabilistic rejection
    rejection_probability = (capacity_used - 0.7) / 0.3
    if random.random() < rejection_probability:
        raise HTTPException(503, "Server at high capacity")
```

## Key Improvements

1. **No More Permanent Lockout**: Counter resets every 5 minutes
2. **Better User Experience**: Not all-or-nothing rejection
3. **Configurable Threshold**: Currently set at 70% to start rejection
4. **Enhanced Metrics**: Added `time_until_reset` and `rejection_threshold` to statistics

## How It Works

### Below 70% Capacity
- All requests accepted
- No rejections

### 70-100% Capacity  
- Graduated rejection probability
- At 70%: 0% rejection
- At 85%: 50% rejection
- At 100%: 100% rejection

### Above 100% Capacity
- All requests rejected (same as before)

### Every 5 Minutes
- Counter resets to 0
- Fresh start for abandoned thread tracking

## Testing the Implementation

To verify it works:

1. **Generate timeouts** to increase abandoned threads
2. **Check graduated response** at different capacity levels
3. **Wait 5 minutes** to see automatic reset
4. **Monitor health endpoint** for new metrics

## Configuration

Currently hardcoded but could be made configurable:
- `reset_interval`: 300 seconds (5 minutes)
- `rejection_start_threshold`: 0.7 (70%)

## Next Steps

If this works well in production, consider:
1. Making intervals configurable via command line
2. Adding metrics/monitoring for rejection rates
3. Possibly implementing Stage 3 (full time-based decay) if needed