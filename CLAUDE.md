# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

**Important**: Always activate the conda environment before running Python commands:
```bash
conda activate llm_behav
```

## Common Development Commands

### Core Experiment Commands
```bash
# Run experiment series with specific configuration
python -m neuro_manipulation.experiment_series_runner --config config/qwen2.5_Series_Prisoners_Dilemma.yaml

# Run individual experiments
python examples/run_option_probability_experiment.py
python examples/run_choice_selection_experiment.py

# Start OpenAI-compatible server with neural hooks (module way)
python -m openai_server --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct --emotion anger

# Start OpenAI-compatible server (backward compatibility)
python init_openai_server.py --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct --emotion anger
```

### Testing Commands
```bash
# Complete compatibility test
./run_complete_test.sh

# Individual component tests
python test_vllm.py
python -m openai_server.tests.test_openai_server
python -m openai_server.tests.test_integrated_openai_server
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

Currently working on:
- Analyzing results from option probability experiments
- Validating experimental design effectiveness

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

- Please use openai_server/manage_servers.py to manage VRAM occupied by the openai_servers, don not use pkill
