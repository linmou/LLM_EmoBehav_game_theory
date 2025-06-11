# Investigating LLM Emotional Reactions in Game Theory Scenarios

[Brief introduction: 1-2 paragraphs about the project, its goals, and the core hypothesis.]

## Table of Contents (Optional, good for longer READMEs)
- [Overview](#overview)
- [Documentation](#documentation)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Data Creation](#1-data-creation)
  - [2. API-based Testing & Prompt Experiments](#2-api-based-testing--prompt-experiments)
  - [3. Activation Steering (Neuro-Manipulation) Experiments](#3-activation-steering-neuro-manipulation-experiments)
- [Directory Structure](#directory-structure) (Optional)
- [Contributing](#contributing) (Optional)
- [License](#license) (Optional)

## Overview
[Expand on the project's hypothesis and goals. Briefly mention the approach to emotion setting, referencing `constants.py`.]

## Documentation
For detailed documentation on project components, methodologies, and specific implementations, please refer to our [main documentation page](docs/index.md) or browse the `docs/` directory.

## Setup
Follow these steps to set up your environment:
1.  **Prerequisites**:
    *   Python 3.x
    *   Conda
2.  **Create and Activate Conda Environment**:
    ```bash
    # conda create -n llm python=3.x (if not already created)
    conda activate llm
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Setup API keys**
    Build an api_configs.py as follows: 

    ```python
    """
    Configuration file containing various API endpoints and configurations for different LLM providers.
    Each configuration contains the necessary parameters to connect to different LLM APIs.
    """

    # OpenAI Official API Configuration
    OAI_CONFIG = {
        "base_url": "https://***",
        "api_key": "sk-**",  # Replace with your API key
    }

    ```

## Usage

The project generally follows three main steps:

### 1. Data Creation
[Brief explanation of data creation purpose]

*   **AG2 based Scenario Creator**:
    ```bash
    python -m data_creation.create_scenario
    ```
    [Briefly what this does and where output goes]

*   **LangGraph-based Scenario Creator**:
    This system offers iterative refinement for scenario generation.
    See [Scenario Creation Graph Documentation](doc/scenario_creation_graph.md) for details.

### 2. API-based Prompt Experiments
    This project also supports prompt-based experiment.

*   **Run Experiment Pipeline**:
    ```bash
    python prompt_experiment.py
    ```
    [Briefly explain "the pipeline", its inputs/outputs]

### 3. Activation Steering (Neuro-Manipulation) Experiments
[Briefly explain what Activation Steering is in this context. Mention it involves techniques like representation engineering to influence model behavior.]

To run a series of experiments with different game and model combinations:
```bash
python -m neuro_manipulation.experiment_series_runner --config [path_to_your_config_file.yaml]
```
Example:
```bash
python -m neuro_manipulation.experiment_series_runner --config config/qwen2.5_Series_Prisoners_Dilemma.yaml
```
Configuration files can be found in the `config/` directory. You can create custom configurations for different experimental setups.
[Link to relevant detailed documentation in `docs/` if available, e.g., for `RepControlVLLMHook`].

## Directory Structure (Optional Example)
- `config/`: Experiment configuration files.
- `data_creation/`: Scripts for generating game scenarios.
- `docs/`: Detailed project documentation.
- `neuro_manipulation/`: Code for activation steering and representation engineering.
- ...

## Contributing (Optional)
[Your contributing guidelines]

## License (Optional)
[Your project license, e.g., MIT, Apache 2.0]