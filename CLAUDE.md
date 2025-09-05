# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

**Important**: Always activate the conda environment before running Python commands:
```bash
conda activate llm_behav
```

**Memory**:
- No need to `conda activate llm_fresh` every time, just ensure conda env is `llm_fresh`
- Use graphiti to update your understanding of the repo when you build new dependencies or find errors in existing graphiti memories
- Be cautious about making too many memory updates to avoid confusion across different memory tracking systems

## Common Development Commands

### Core Experiment Commands
```bash
# Run experiment series with specific configuration
python -m neuro_manipulation.experiment_series_runner --config config/qwen2.5_Series_Prisoners_Dilemma.yaml

# Run individual experiments
python examples/run_option_probability_experiment.py
python examples/run_choice_selection_experiment.py
```

### Starting OpenAI-Compatible Server

**IMPORTANT**: The server requires BOTH `--model` (path) AND `--model_name` (API name) arguments!

#### Method 1: Module approach (recommended)
```bash
# Basic start (make sure conda env llm_fresh is active)
python -m openai_server \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B-anger \
    --emotion anger

# Start in background with logging
nohup python -m openai_server \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B-anger \
    --emotion anger > server.log 2>&1 &
```

#### Method 2: Init script (backward compatibility)
```bash
python init_openai_server.py \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B-anger \
    --emotion anger
```

#### Starting from a bash script or non-interactive shell
```bash
# When conda activate doesn't work directly (e.g., in scripts)
bash -c "source /usr/local/anaconda3/etc/profile.d/conda.sh && \
    conda activate llm_fresh && \
    python -m openai_server \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B-anger \
    --emotion anger"
```

#### Common Server Arguments
- `--model`: Path to the model directory (REQUIRED)
- `--model_name`: Name for the API endpoint (REQUIRED) 
- `--emotion`: Emotion to activate (default: anger)
- `--port`: Port to run on (default: 8000)
- `--gpu_memory_utilization`: GPU memory usage (default: 0.90)
- `--max_num_seqs`: Max concurrent sequences (default: 64)

#### Graceful Degradation Arguments (Phase 4.1)
- `--request_timeout`: Request timeout in seconds (default: 60)
- `--max_queue_size`: Max requests in queue (default: 50)
- `--max_concurrent_requests`: Max concurrent processing (default: 3)
- `--queue_rejection_threshold`: Queue rejection threshold 0.0-1.0 (default: 0.8)
- `--reset_interval`: Abandoned thread reset interval in seconds (default: 300)
- `--vllm_rejection_threshold`: Start probabilistic rejection 0.0-1.0 (default: 0.7)

### Testing Commands
```bash
# Complete compatibility test
./run_complete_test.sh

# Individual component tests
python test_vllm.py
python -m openai_server.tests.test_openai_server
python -m openai_server.tests.test_integrated_openai_server

# Thinking mode testing
python tests/thinking_mode/test_thinking_complete.py
```

### Scenario Generation
```bash
# Generate scenarios using LangGraph (recommended)
python -m data_creation.create_scenario_langgraph

# Generate using AutoGen
python -m data_creation.create_scenario
```

### Documentation
```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Project Architecture

This is a research project investigating **LLM emotional reactions in game theory scenarios**. The core hypothesis explores whether emotional states can be artificially induced in LLMs and how these affect strategic decision-making.

### Key Components

**Neural Manipulation Framework** (`/neuro_manipulation/`)
- **Representation Engineering (RepE)**: Emotion activation vector extraction and real-time neural intervention
- **vLLM Hook System**: `RepControlVLLMHook` and `SequenceProbVLLMHook` for behavioral manipulation
- **Experiment Classes**: `OptionProbabilityExperiment`, `ChoiceSelectionExperiment` for measuring behavioral changes

**Game Theory Engine** (`/games/`)
- Abstract base classes: `Game`, `GameScenario`, `BehaviorChoices`
- Specific implementations: Prisoner's Dilemma, Battle of Sexes, Trust Game, etc.
- Payoff matrices with mathematical foundations
- Game classification: Type (Simultaneous/Sequential) and Symmetry (Symmetric/Asymmetric)

**Configuration System** (`/config/`)
- YAML-based experiment configurations
- Model configurations for Qwen series
- Game-specific parameter settings

**Data Pipeline** (`/data_creation/`)
- LangGraph-based scenario generation with iterative refinement
- AutoGen multi-agent scenario creation
- Emotion-based stimulus categorization

**OpenAI Server Module** (`/openai_server/`)
- OpenAI-compatible FastAPI server with emotion control integration
- Modular architecture with proper Python package structure
- Comprehensive test suite and backward compatibility
- Production-ready deployment capabilities

### Emotion System

The project uses a six-emotion classification system defined in `constants.py`:
- Primary emotions: anger, happiness, sadness, disgust, fear, surprise
- Intelligent string matching with prefix handling
- Intensity scaling for graduated responses

### Key Design Patterns

1. **Modular Game Framework**: Extensible game implementations with unified interfaces
2. **Neural Intervention Pipeline**: Extract emotion vectors → Apply vLLM hooks → Measure behavioral changes
3. **Experiment Factory**: Configuration-driven experiment setup with automated result collection
4. **Multi-Modal LLM Integration**: Support for both local models (vLLM) and OpenAI API

## Important Files and Configuration

**Core Configuration**
- `constants.py`: Emotion classification system and core enums
- `requirements.txt`: Python dependencies (PyTorch, vLLM, Transformers, etc.)
- `api_configs.py`: API endpoint configurations (user-created, contains API keys)

**Experiment Tracking**
- Results stored in `/results/` organized by experiment type and model
- Comprehensive logging with timestamps and statistical analysis
- JSON and CSV output formats

## Current Development Status

Based on `tasks.md`, the project has completed:
- ✅ MkDocs documentation system
- ✅ Sequence probability vLLM hook implementation
- ✅ Emotion-context defection probability study framework
- ✅ OpenAI-compatible server with neural hooks


## Organized File Structure

### Thinking Mode Components
- **Testing**: `tests/thinking_mode/test_thinking_complete.py` - Comprehensive thinking mode validation
- **Experiments**: `examples/thinking_mode/run_qwen3_thinking_mode_experiment.py` - Full experiment runner
- **Analysis**: `result_analysis/thinking_mode/analyze_thinking_mode_ratio.py` - Result analysis tools
- **Documentation**: `docs/experiments/thinking_mode/Qwen3_Thinking_Mode_Implementation_Report.md`

## Model Support

The project primarily uses:
- **Qwen model series** (local inference via vLLM)
- **OpenAI API models** (for comparison studies)
- **CUDA support** required for local model inference

## Testing and Validation

The project includes comprehensive testing:
- vLLM compatibility tests
- OpenAI server integration tests
- Statistical analysis validation
- Tensor parallel consistency checks

## OpenAI Server Management
- You can use openai_server/manage_servers.py to see existing servers or kill them.
- Please use openai_server/manage_servers.py to manage VRAM occupied by the openai_servers, don not use pkill

### Server Management Commands
```bash
# View running servers
python openai_server/manage_servers.py

# Kill all servers and orphaned processes
echo -e "c\ny" | python openai_server/manage_servers.py

# Check server health
curl http://localhost:8000/health | python -m json.tool
```

### Common Server Issues & Solutions

1. **"error: the following arguments are required: --model_name"**
   - Solution: Add BOTH `--model` and `--model_name` arguments

2. **"CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'"**
   - Solution: Use the bash script method shown above with `source .../conda.sh`

3. **"Server at capacity due to ongoing requests"** 
   - This means graceful degradation is working! Too many abandoned threads
   - Solution: Restart the server to clear abandoned threads
   - **Production Fix**: Use `production_async_vllm_wrapper.py` which has automatic recovery
   - **Note**: Tests 2 & 3 in stress suite fail due to this (accumulated abandoned threads)

4. **Server won't start or hangs**
   - Check the log file: `tail -50 server.log`
   - Make sure no other server is using port 8000
   - Use manage_servers.py to clean up old processes

```
```