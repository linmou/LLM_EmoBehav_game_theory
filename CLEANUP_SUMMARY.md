# Test Files Cleanup Summary

## Changes Made

### 1. OpenAI Server Tests Organization
- **Moved to `openai_server/tests/`:**
  - `test_server_management.py`
  - `test_server_management_complete.py`
  - `test_manage_servers.py`
  - `test_server_connectivity.py`

- **Moved to `openai_server/`:**
  - `manage_servers.py` (server management utility)

- **Moved to `openai_server/examples/`:**
  - `sanity_check_example.py`

- **Removed (were backward compatibility wrappers):**
  - `test_openai_server.py` (root)
  - `test_integrated_openai_server.py` (root)

### 2. Experiment Scripts Organization
- **Created `experiments/neural_emotion/` and moved:**
  - `run_300_efficient.py`
  - `run_300_scenario_experiment.py`
  - `run_300_simple.py`
  - `run_50_scenario_comparison.py`
  - `run_anger_300.py`
  - `run_neural_emotion_experiment.py`

### 3. General Test Files
- **Created `tests/` and moved:**
  - `run_complete_compatibility_test.py`

### 4. Neural Manipulation Tests
- **Moved to `neuro_manipulation/tests/`:**
  - `test_repcontrol_api.py`
  - `test_vllm.py`
  - `test_vllm_repcontrol_api.py`

### 5. Log Files Cleanup
- **Created `logs/server/` and moved:**
  - All `*server*.log` files
  - `nohup.out` (if present)

## Current Structure

```
openai_server/
├── __init__.py
├── __main__.py
├── server.py
├── manage_servers.py
├── examples/
│   └── sanity_check_example.py
└── tests/
    ├── __init__.py
    ├── test_integrated_openai_server.py
    ├── test_openai_server.py
    ├── test_manage_servers.py
    ├── test_server_connectivity.py
    ├── test_server_management.py
    └── test_server_management_complete.py

experiments/
└── neural_emotion/
    └── [various run_*.py scripts]

neuro_manipulation/
└── tests/
    ├── test_repcontrol_api.py
    ├── test_vllm.py
    └── test_vllm_repcontrol_api.py

logs/
└── server/
    └── [various server log files]
```

## Notes
- Kept `init_openai_server.py` in root as it's a backward compatibility wrapper
- All test files are now properly organized by module
- Experiment scripts are separated from core code
- Log files are organized in a dedicated directory