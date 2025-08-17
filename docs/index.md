# LLM Emotional Behavior in Game Theory

This project investigates LLM emotional reactions in game theory scenarios through neural manipulation and representation engineering.

## Quick Start

* `mkdocs serve` - Start the live-reloading docs server
* `mkdocs build` - Build the documentation site
* See [CLAUDE.md](../CLAUDE.md) for development setup and common commands

# Project Documentation

This directory contains documentation for the different modules within the LLM_EmoBehav_game_theory project.

## Core Framework

### Experiments
*   **[Qwen3 Thinking Mode](./experiments/qwen3_thinking_mode.md)** - Complete guide to implementing and configuring thinking mode for Qwen3 models
*   **[Experiment Series](./reference/experiment_series_README.md)** - Documentation for running experiment series
*   **[Token Management](./reference/token_management.md)** - Token configuration, analysis, and optimization strategies

### Neural Manipulation
*   **[Representation Engineering (RepE)](./code_readme/neuro_manipulation/repe/README.md)** - Extracting emotion activation directions
*   **[Sequence Probability Fix](./code_readme/neuro_manipulation/repe/README_sequence_prob_fix.md)** - vLLM v1 compatible sequence probability calculation
*   **[vLLM Hook Implementation](./reference/vllm_hook_implementation.md)** - RepControlVLLMHook for representation control

### Models and Prompts
*   **[Prompt Format](./reference/prompt_format.md)** - Multi-model prompt formatting system
*   **[Prompt Wrapper](./reference/prompt_wrapper.md)** - High-level prompt management
*   **[Model Layer Detector](./reference/model_layer_detector.md)** - Automatic layer detection for different models
*   **[vLLM Compatibility](./reference/vllm_compatibility.md)** - vLLM integration with representation engineering

## Game Theory Components

### Scenario Generation
*   **[LangGraph Implementation](./code_readme/data_creation/scenario_creation/langgraph_creation/README.md)** - Graph-based scenario generation
*   **[Enhanced Scenario Generation](./code_readme/scenario_generation_restart.md)** - Production-ready generation with error recovery

### Game Framework
*   **[Constants Module](../README_constants.md)** - Emotion and game theory classification system
*   **[Payoff Matrices](./reference/payoff_matrices.md)** - Mathematical foundations of game payoffs
*   **[Game Tree](./reference/game_tree.md)** - Game structure documentation

## Analysis and Results

### Behavioral Analysis
*   **[Emotion Analysis](./reference/emotion_analysis.md)** - Emotion detection and classification
*   **[Option Pattern Analysis](./code_readme/result_analysis/option_pattern_analysis_link.md)** - Linguistic patterns in decisions
*   **[Statistical Engine](./reference/statistical_engine.md)** - Statistical analysis framework

### Experiment Management
*   **[Experiment Time Tracking](./reference/ExperimentTimeTracking.md)** - Performance monitoring
*   **[Experiment Report Naming](./reference/experiment_report_naming.md)** - Result organization conventions

