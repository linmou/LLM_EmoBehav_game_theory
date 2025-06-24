# Project Tasks

## Documentation Setup

- [x] **VAN Mode: Initiate Documentation Webpage** (Initial setup and site build complete)
  - [x] Choose and set up a documentation tool (MkDocs).
  - [x] Inspect existing documentation in the `doc` folder.
  - [x] Migrate existing `doc` content to the new documentation structure.
  - [x] **IMPLEMENT Mode: By comparing with the relevant code, review existing documentation content about the correctness of contents and the coverage .** (Commencing)
    - [x] Review `docs/reference/model_layer_detector.md`
    - [x] Review `docs/reference/vllm_hook_implementation.md`
    - [x] Review `docs/reference/emotion_analysis.md` 
    - [x] Review `docs/reference/vllm_compatibility.md`
    - [x] Review and expand `docs/reference/statistical_engine.md` (Identified as stub, recreated, further review pending if substantive content is added later)
    - [x] Review `docs/reference/experiment_report_naming.md`
    - [x] Review `docs/reference/experiment_series_README.md`
    - [x] Review `docs/reference/prompt_format.md`
    - [x] Review `docs/reference/prompt_wrapper.md`
    - [x] Review `docs/reference/model_download_management.md`
    - [x] Review `docs/reference/payoff_matrices.md`
    - [x] Review `docs/reference/scenario_creation_graph.md` (and ensure `scenario_creation_graph.png` is handled)
  - [x] Create a guide on how to build and maintain the documentation.
  - [x] Build and verify the documentation website.
  - [x] Address `mkdocs build` warnings.
    - [x] Resolve `README.md` vs `index.md` conflict in `docs` directory.
    - [x] Fix broken links in `docs/index.md`.
    - [x] Fix broken links in `docs/reference/emotion_analysis.md`.
    - [x] Fix broken links in `docs/reference/model_layer_detector.md`. 

## Neural Manipulation Development

- [x] **VAN Mode: Build Sequence Probability vLLM Hook** (Level 1 - Direct Implementation)
  - [x] **IMPLEMENT Mode: Create SequenceProbVLLMHook class with tensor parallel support**
    - [x] Design hook function to capture logits from language model head
    - [x] Implement RPC functions for tensor parallel communication
    - [x] Create main SequenceProbVLLMHook class with get_log_prob method
    - [x] Add proper logit aggregation across tensor parallel ranks
    - [x] Include example usage and testing script
    - [x] Write documentation and unit tests
    - [x] Create README.md for the sequence probability functionality 


## Emotion Intervention and Context Study

- [x] **VAN Mode: Build Emotion-Context Defection Probability Study** (Level 2 - Complex Analysis)
  - [x] **PLAN Mode: Design dual-factor experiment framework**
    - [x] Define emotion intervention (anger as an example) effect on defect behavior probability
    - [x] Design scenario description context manipulation (with/without description)
    - [x] Plan statistical analysis for interaction effects between emotion and context
    - [x] Define experimental conditions and control groups
  - [x] **IMPLEMENT Mode: Create new Experiment and Dataset classes**
    - [x] Create `OptionProbabilityExperiment` class to measure probabilities for all available options
    - [x] Create `ContextManipulationDataset` as a subclass of `GameScenarioDataset`
    - [x] In `ContextManipulationDataset`, modify `__getitem__` to conditionally include scenario description
    - [x] Integrate `SequenceProbVLLMHook` into `OptionProbabilityExperiment` to measure probabilities for all options
    - [x] Implement 2x2 factorial design (Emotion x Context) in the new experiment class
    - [x] Add statistical analysis for probability data, including interaction effects
    - [x] Write comprehensive unit tests for both new classes
    - [x] Write documentation for the new experiment design
  - [x] #task-1 Sanity check the `OptionProbabilityExperiment`
    - **Goal**: Run the `OptionProbabilityExperiment` with a small data sample to ensure it runs end-to-end without crashing.
    - **Outcome**: The experiment now runs to completion. The statistical analysis was made robust to handle cases where no probabilities can be calculated for any of the prompts, which was causing the experiment to crash. The temporary runner script and configuration file were deleted.
  - [ ] #task-2 Analyze the results of the `OptionProbabilityExperiment`
    - **Goal**: Analyze the results of the `OptionProbabilityExperiment` to understand how emotion and context affect the model's choices.
    - **Outcome**: TBD
    - [ ] Run sanity check with small sample to verify experimental design
    - [ ] Validate that sequence probabilities are accurately measured for all behavior choices
    - [ ] Verify statistical analysis correctly identifies interaction effects
    - [ ] Test edge cases and ensure robust error handling

## LLM API Server Development

- [x] **VAN Mode: Enable vLLM Hook Server as OpenAI Server** (Level 1 - Direct Implementation)
  - [x] **IMPLEMENT Mode: Create OpenAI-compatible server with vLLM hook integration**
    - [x] Design init_openai_server.py script to start vLLM server with emotion hooks
    - [x] Implement command-line interface for emotion selection and model configuration
    - [x] Create OpenAI-compatible API endpoints that integrate RepControlVLLMHook
    - [x] Add proper error handling and logging for server operations
    - [x] Create test case to verify server initiation and client communication
    - [x] Write documentation for server setup and usage
    - [x] Validate OpenAI client compatibility with hooked model responses