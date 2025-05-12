# Documentation Folder

[→ Documentation Index](index.md)

All documentation for this project has been consolidated into the `doc` folder. The previous `docs` folder has been merged and removed as of [date of merge].

## Structure
- All markdown files and images related to documentation are now in `doc/`.
- Please update any scripts or references to use `doc/` instead of `docs/`.

## Contents
- Experiment reports
- Model compatibility
- Scenario creation
- Game theory documentation
- Emotion analysis
- And more...

For a full list of documentation files, see the contents of this folder.

# Project Documentation

This directory contains documentation for the different modules within the LLM_EmoBehav_game_theory project.

## Modules

*   **Scenario Creation:**
    *   [LangGraph Implementation](./../data_creation/scenario_creation/langgraph_creation/README.md): Details on the graph-based process for generating game theory scenarios.
*   **Games:**
    *   See the `docs/games/` directory for specific game configurations and details.

*   **Representation Reading (RepE)**: 
    *   [Extracting Emotion Activation Directions](./../neuro_manipulation/repe/README.md)

*   **Emotion Analysis**:
    *   [Emotion Analysis](./emotion_analysis.md) - Documentation for emotion analysis

*   **Model Layer Detector**:
    *   [Model Layer Detector](./model_layer_detector.md) - Documentation for model layer detector

*   **Prompt Format**:
    *   [Prompt Format](./prompt_format.md) - Documentation for prompt format

*   **Prompt Wrapper**:
    *   [Prompt Wrapper](./prompt_wrapper.md) - Documentation for prompt wrapper

*   **Experiment Series**:
    *   [Experiment Series](./experiment_series_README.md) - Documentation for experiment series



*   **vLLM Integration**:
    *   [vLLM Compatibility](./vllm_compatibility.md) - Documentation for vLLM compatibility with representation engineering
    *   [vLLM Hook Implementation](./vllm_hook_implementation.md) - Explains the `RepControlVLLMHook` for representation control using vLLM hooks.

*   ... (Add links to other documentation as needed) 