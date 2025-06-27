# Neuro-Manipulation Toolkit

This toolkit provides functionalities for exploring and manipulating the internal representations of language models, particularly focusing on emotions. It allows users to:

1.  **Extract Emotion Representations**: Identify and extract "steering vectors" that represent specific emotions (e.g., anger, joy) within the model\'s hidden states.
2.  **Run Controlled Experiments**: Conduct experiments by injecting these emotion steering vectors into a model during inference to observe their impact on behavior, such as decision-making in game scenarios.


## Core Components and Workflow

The toolkit revolves around a few key components and processes:

### 1. Extracting Emotion Activation Directions (Steering Vectors)

This process uses the `RepReadingPipeline` and the `all_emotion_rep_reader` utility function to discover directions in a model\'s activation space that correspond to specific emotions.

**Conceptual Steps:**

1.  **Data Preparation**:
    *   Organize text data into a dictionary where keys are emotion labels (e.g., \'anger\', \'joy\').
    *   Each emotion maps to \'train\' and \'test\' lists of text examples.
    *   For directional methods like \'pca\' or \'rotation\', provide contrastive pairs (e.g., \[negative\_example, positive\_example,...]).

2.  **Pipeline Initialization**:
    *   Set up a `RepReadingPipeline` with a transformer model and tokenizer.
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    # Assuming RepReadingPipeline is correctly importable
    from neuro_manipulation.repe import RepReadingPipeline

    model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Example
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer, pipeline_class=RepReadingPipeline)
    ```

3.  **Parameter Definition**:
    *   `emotions`: List of target emotions.
    *   `hidden_layers`: Model layers to extract hidden states from.
    *   `rep_token`: Token index for representation analysis (e.g., -1 for the last token).
    *   `n_difference`: Number of differences between input pairs (for methods like \'pca\').
    *   `direction_method`: Strategy like \'pca\', \'rotation\', or \'clustermean\'.
    *   `save_path`: Optional path to save the `RepReader` objects.

4.  **Execution**:
    *   Call `all_emotion_rep_reader` with the data, pipeline, and parameters.
    ```python
    # Assuming all_emotion_rep_reader is correctly importable
    from neuro_manipulation.utils import all_emotion_rep_reader

    # emotions, hidden_layers, data, etc. are defined
    emotion_rep_readers = all_emotion_rep_reader(
        data=data,
        emotions=emotions,
        rep_reading_pipeline=rep_reading_pipeline,
        # ... other parameters
    )
    ```

5.  **Results**:
    *   The function returns a dictionary containing trained `RepReader` objects for each emotion.
    *   Each `RepReader` stores the `directions` (layer-wise emotion vectors) and `direction_signs`.
    *   It also includes layer-wise accuracy and the arguments used.

**Internal Mechanics:**

*   The `RepReadingPipeline` extracts hidden states for specified tokens and layers.
*   It then uses a `DIRECTION_FINDER` (e.g., PCA) on these hidden states (or their differences) to compute the direction vectors.
*   The `all_emotion_rep_reader` function orchestrates this for multiple emotions, including a validation step to assess the quality of the learned directions using test data.

The output is a set of vectors (one per layer per emotion) representing the identified emotional directions in the model\'s activation space.

*(For more details, refer to `docs/code_readme/neuro_manipulation/repe/README_extract_emotion_repe.md`)*

### 2. Emotion Game Experiment Logic (`emotion_game_experiment.py`)

The `EmotionGameExperiment` class is designed to investigate how injecting these learned emotion steering vectors affects a language model\'s behavior in game-like scenarios that require decision-making.

**Experiment Workflow:**

1.  **Initialization (`__init__`)**:
    *   Sets up logging.
    *   Loads configurations for the RepE engine, the experiment itself, and the game.
    *   Initializes the model and tokenizer (supports both Hugging Face Transformers and vLLM).
        *   Crucially, it loads the pre-computed `emotion_rep_readers` (steering vectors) using `load_emotion_readers`.
    *   Determines the hidden layers to be targeted for manipulation.
    *   Initializes a `rep_control_pipeline` (either `rep-control` for standard Transformers or `rep-control-vllm` for vLLM) which will be responsible for applying the steering vectors during inference. This pipeline is configured with the control layers, block name (e.g., \'decoder\_block\'), and control method.
    *   Sets up a `GameReactPromptWrapper` to format prompts for the game scenarios and parse model responses.
    *   Prepares an output directory for saving results and configurations.

2.  **Data Loading (`build_dataloader`)**:
    *   Creates a `GameScenarioDataset` which provides game scenarios, descriptions, and options.
    *   The dataset uses the `GameReactPromptWrapper` to format the input prompts for the LLM.
    *   A PyTorch `DataLoader` is used to batch these scenarios.

3.  **Running the Experiment (`run_experiment`)**:
    *   Iterates through each specified `emotion` and each `intensity` (coefficient) for applying the emotion.
    *   For each emotion, retrieves the corresponding `RepReader` (containing the steering vectors).
    *   For each intensity, calls `_infer_with_activation`.
    *   Also runs a "Neutral" condition (zero intensity) as a baseline.
    *   Saves all results.

4.  **Inference with Activation Control (`_infer_with_activation`)**:
    *   Constructs the `activations` dictionary. This maps each target layer index to a tensor representing `coefficient * direction_vector * direction_sign`. This is the actual "steering" signal.
    *   Calls `_forward_dataloader` to process the game scenarios with these activations.

5.  **Forwarding Data through Model (`_forward_dataloader`)**:
    *   This method processes batches of game scenarios.
    *   It uses a worker thread (`pipeline_worker`) and a `Queue` to manage the flow of data and results, allowing for asynchronous generation and post-processing.
    *   For each batch:
        *   It repeats samples within the batch if `self.repeat > 1`.
        *   It calls the `self.rep_control_pipeline` with the prompts and the prepared `activations`. This is where the model generates responses *while under the influence* of the injected emotion.
        *   The generation parameters (temperature, max tokens, etc.) are taken from the experiment configuration.
    *   A `ThreadPoolExecutor` is used to parallelize the post-processing of the generated outputs.

6.  **Post-Processing Batch Results (`_post_process_batch`)**:
    *   For each generated output in a batch:
        *   Extracts the raw generated text.
        *   Calls `_post_process_single_output` to parse the rationale and decision from the text.
        *   Formats and collects the results, including emotion, intensity, scenario details, input prompt, raw output, parsed rationale, decision, chosen option category, and repeat number.

7.  **Post-Processing Single Output (`_post_process_single_output`)**:
    *   Attempts to extract the "rationale" and "decision" from the model\'s generated text using regex patterns.
    *   If regex fails, it falls back to using an LLM (e.g., GPT-4o-mini via `_get_option_id_from_llm`) to interpret the output and identify the chosen option.
    *   Determines the `option_id` corresponding to the extracted decision.

8.  **Saving Results (`_save_results`)**:
    *   Saves the collected experimental data to JSON and CSV files.
    *   Calls `_run_statistical_analysis` to perform statistical tests on the results.

9.  **Statistical Analysis (`_run_statistical_analysis`)**:
    *   Uses an external `analyze_emotion_and_intensity_effects` function (from `statistical_engine`) to analyze the impact of different emotions and intensities on the model\'s decisions.
    *   Saves these statistical results.

10. **Sanity Check (`run_sanity_check`)**:
    *   Provides a way to run a small version of the experiment (e.g., 10 samples, one emotion, one intensity) to quickly validate the setup.

This experiment systematically tests how nudging the model\'s internal state with emotion-specific vectors influences its choices in defined scenarios.

### 3. Inference with vLLM Hooks (`RepControlVLLMHook`)

When using vLLM for faster inference, the `RepControlVLLMHook` class (detailed in `docs/reference/vllm_hook_implementation.md`) enables applying representation control.

**Mechanism:**

1.  **Initialization**:
    *   A `RepControlVLLMHook` instance is created with the vLLM `LLM` object, tokenizer, target layer indices, the module name within the layer to hook (e.g., `"decoder_block"`), and the control method.
    *   It uses vLLM\'s `collective_rpc` (Remote Procedure Call) to register a forward hook (`hook_fn_rep_control`) on the specified module of the model on each vLLM worker process.
    *   Initially, the hook is dormant as no control state is set.

2.  **Controlled Generation (`__call__` method of the hook, or via `rep-control-vllm` pipeline):**
    *   **Set State**: Before running inference, if `activations` (the steering vectors scaled by intensity) are provided, an RPC call (`_set_controller_state_on_worker_rpc`) is made to each worker. This attaches the control parameters (steering vector for that layer, mask, operator, etc.) as a `_rep_control_state` attribute to the hooked module on that worker.
    *   **Run Inference**: Standard `model.generate()` is called (vLLM handles the distributed inference).
        *   When the forward pass on a worker reaches a hooked module, `hook_fn_rep_control` executes.
        *   The hook checks for the `_rep_control_state`.
        *   If present, it retrieves the control tensor and other parameters and modifies the module\'s output (e.g., by adding the steering vector).
        *   If no state is present, it passes the original output through.
    *   **Reset State**: After generation, another RPC call (`_reset_controller_state_on_worker_rpc`) removes the `_rep_control_state` from the modules on all workers. This ensures subsequent inference calls are not affected.

3.  **Hook Function (`hook_fn_rep_control`)**:
    *   This is the core logic that applies the representation modification (e.g., adding the control vector to the module\'s output), similar to how `WrappedBlock.forward` would work in non-vLLM setups. It handles tensor manipulations, masking, and applying the operator.

**Advantages:**

*   Integrates with vLLM without modifying its core model structure.
*   Uses vLLM\'s RPC for managing distributed state (the control vectors).

**Considerations:**

*   RPC calls introduce some overhead.
*   State management complexity relies on the RPC mechanism.
*   Hooks might interact with vLLM\'s internal optimizations; `enforce_eager=True` might be needed for the vLLM `LLM` object.

*(For more details, see `docs/reference/vllm_hook_implementation.md`)*

## How to Use

1.  **Environment Setup**: Ensure all dependencies, including `transformers`, `torch`, `vllm` (if used), and other packages listed in `emotion_game_experiment.py` are installed.
2.  **Configuration**:
    *   Prepare configuration files (YAML) for:
        *   **RepE Engine (`repe_eng_config`)**: Specifies model details, layers for control, block names, control method, paths to emotion datasets for `RepReader` training, etc.
        *   **Experiment (`exp_config`)**: Defines emotions to test, intensity coefficients, LLM generation parameters, output directories, system messages for prompts.
        *   **Game (`game_config`)**: Specifies the game data path, decision class structure for parsing outputs.
3.  **Prepare Emotion Representations(Optional)**:
    *   If not already done, run the process described in "Extracting Emotion Activation Directions" to generate and save `RepReader` objects for your target emotions and model. Ensure the `load_emotion_readers` function in `EmotionGameExperiment` can access these.
    *   The `EmotionGameExperiment` includes the logic that generates and saves `RepReader` objects if not exists. 
4.  **Run an Experiment**:
    *   Instantiate `EmotionGameExperiment` with the loaded configurations.
    *   Call the `run_experiment()` method to execute the full experiment.
    *   Alternatively, call `run_sanity_check()` for a quick test.

## Directory Structure (Key Files)

*   `neuro_manipulation/`
    *   `experiments/`
        *   `emotion_game_experiment.py`: Main script for running emotion manipulation experiments.
    *   `repe/`
        *   `README_extract_emotion_repe.md`: (Reference) Detailed guide on extracting emotion vectors.
        *   `pipelines.py`: Contains `RepReadingPipeline` and `RepControlPipeline` (and their vLLM variants).
        *   `rep_control_vllm_hook.py`: (Reference) Implements the vLLM hook mechanism.
    *   `datasets/`
        *   `game_scenario_dataset.py`: Defines the dataset for game scenarios.
    *   `utils/`
        *   `all_emotion_rep_reader` (likely here or in a similar utility script): Function to train all emotion readers.
        *   Other helper functions.
    *   `model_utils.py`: Utilities for model and tokenizer setup, loading emotion readers.
    *   `prompt_wrapper.py`: For formatting prompts and structuring responses.
    *   `README.md`: (This file) Overview of the toolkit.
*   `docs/`
    *   `code_readme/neuro_manipulation/repe/README.md`: (Linked from `current_file`) Likely a duplicate or older version of `README_extract_emotion_repe.md`.
    *   `reference/vllm_hook_implementation.md`: (Linked from `current_file`) Detailed explanation of the vLLM hook.

This toolkit provides a comprehensive framework for investigating and influencing LLM behavior through targeted manipulations of their internal emotional representations. 