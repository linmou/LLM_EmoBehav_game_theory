# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

# Project Documentation

This directory contains documentation for the different modules within the LLM_EmoBehav_game_theory project.

## Modules

*   **Scenario Creation:**
    *   [LangGraph Implementation](./code_readme/data_creation/scenario_creation/langgraph_creation/README.md): Details on the graph-based process for generating game theory scenarios.
    *   [🔄 Enhanced Scenario Generation with Restart Capabilities](./code_readme/scenario_generation_restart.md): Production-ready scenario generation with automatic restart, timeout handling, and comprehensive error recovery.
*   **Games:**
    *   See the `docs/games/` directory for specific game configurations and details.

*   **Representation Reading (RepE)**: 
    *   [Extracting Emotion Activation Directions](./code_readme/neuro_manipulation/repe/README.md)

*   **Emotion Analysis**:
    *   [Emotion Analysis](./reference/emotion_analysis.md) - Documentation for emotion analysis

*   **Model Layer Detector**:
    *   [Model Layer Detector](./reference/model_layer_detector.md) - Documentation for model layer detector

*   **Prompt Format**:
    *   [Prompt Format](./reference/prompt_format.md) - Documentation for prompt format

*   **Prompt Wrapper**:
    *   [Prompt Wrapper](./reference/prompt_wrapper.md) - Documentation for prompt wrapper

*   **Experiment Series**:
    *   [Experiment Series](./reference/experiment_series_README.md) - Documentation for experiment series

*   **Behavior Analysis**:
    *   [Behavior Analyzer Script](../../README.md): Script to analyze predicate distribution in behavior choices using Azure GPT-4o.

*   **vLLM Integration**:
    *   [vLLM Compatibility](./reference/vllm_compatibility.md) - Documentation for vLLM compatibility with representation engineering
    *   [vLLM Hook Implementation](./reference/vllm_hook_implementation.md) - Explains the `RepControlVLLMHook` for representation control using vLLM hooks.

*   ... (Add links to other documentation as needed)
