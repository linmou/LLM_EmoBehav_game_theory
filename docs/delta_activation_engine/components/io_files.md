# IO Helpers (`delta_activation_engine/io/files.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Minimal persistence layer for vectors and metadata used by both pipelines. Ensures directories exist before writes.

## Implementation Walkthrough
- `ensure_dir(path)`: `os.makedirs(path, exist_ok=True)`.
- `save_npz_vector(path, vector)`: ensures parent directory, saves compressed numpy array under key `vector`.
- `save_json(path, payload)`: ensures parent directory, writes JSON with UTF-8 encoding and 2-space indentation.

## Key Logic
- Centralizes directory creation to avoid scattered `os.makedirs` calls.
- Uses `np.savez_compressed` to keep output size small while retaining float precision.

## Dependencies
- Standard library `os`, `json`; numpy for serialization.

## Potential Issues / Gaps
- No checksum or atomic write; partial files are possible on interruption.
- `save_npz_vector` assumes array is small enough for memory; no streaming.

## Usage Example
```python
import numpy as np
from delta_activation_engine.io.files import save_npz_vector, save_json

vec = np.arange(4, dtype=np.float32)
save_npz_vector("/tmp/deltas/emotion=anger_int=1.0.npz", vec)
save_json("/tmp/deltas/meta.json", {"shape": vec.shape})
```
