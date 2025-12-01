# Task-Similarity PD Defection - Terminology Glossary

This glossary collects the main concepts used in `auto_experiments.task_similarity`. It is aimed at someone reading or extending the PD defection experiments.

## Core Game-Theory Concepts

- **Prisoner's Dilemma (PD)**  
  A two-player game with two options: **cooperate** or **defect**. Defection gives a higher individual payoff irrespective of the other player's choice, but mutual cooperation is socially optimal.

- **Defection**  
  The option in PD where the agent chooses a selfish payoff, typically harming the partner. In this codebase, defection is the **positive** class when training activation directions.

- **Cooperation**  
  The option where the agent sacrifices some payoff for mutual benefit. It is the **negative** class in the PD defection direction.

## Data Structures

- `PairMeta` (`pd_prompt_builder.PairMeta`)  
  Metadata for a single PD scenario: option texts (`opt_a`, `opt_b`), which label (`"A"`/`"B"`) corresponds to **defection** or **cooperation**, and the textual description.

- `PromptPair` (`pd_prompt_builder.PromptPair`)  
  Holds two concrete PD prompts for the same scenario:  
  - `positive`: prompt whose Assistant answer chooses defection.  
  - `negative`: prompt whose Assistant answer chooses cooperation.  
  Both share the same randomized option ordering, so the only difference is the Assistant's choice.

- `PDPairBundle` (`pd_data.PDPairBundle`)  
  A container for:  
  - `pairs`: all PD `PromptPair` objects from the JSON source.  
  - `train_pairs`: subset used for representation learning.  
  - `test_pairs`: disjoint subset used for validation and behavior evaluation.

- **RepReader dataset** (`pd_data.build_repreader_dataset`)  
  A simple dict format expected by `neuro_manipulation`'s RepReading pipeline:  
  - `data`: list of prompt strings; layout is `[pos0, neg0, pos1, neg1, ...]`.  
  - `labels`: list of `[1, 0]` per pair, marking defection as the positive example.

- `PDActivationSpec` (`run_pd_defection_pd_behavior.PDActivationSpec`)  
  Configuration for re-using a previously trained PD defection vector:  
  - `pd_result_dir`: directory of a `run_pd_defection_experiment` run.  
  - `layer`: layer index for the vector.  
  - `vector_path`: path to the steering vector (relative to `pd_result_dir` if needed).  
  - `span_mode`: representation span (usually `"option"`).  
  - `pd_best_layer`, `pd_best_accuracy`: summary metrics from training.  
  - `pd_seed`, `pd_max_pairs`: training split parameters (used to align benchmark data).

## Representations and Vectors

- **Assistant span**  
  In PD prompts, the answer text starts at the `"Assistant:"` marker. Two spans are commonly used:  
  - `"assistant"` span: from `"Assistant:"` through `"LABEL) ANSWER"`.  
  - `"option"` span: only the `"LABEL) ANSWER"` portion, excluding `"Assistant:"`.

- `collect_answer_means` (`pd_hidden_extractor.collect_answer_means`)  
  Given prompts and a list of transformer layers, this function:  
  1. Finds the token index where the desired span starts (assistant or option).  
  2. Runs the model with `output_hidden_states=True`.  
  3. Mean-pools hidden states over the span tokens for each layer.  
  It returns a dict `layer -> (num_prompts, hidden_dim)` of pooled vectors.

- **Defection vector / activation direction**  
  A 1-D vector in hidden-state space that points from cooperation representations towards defection representations. There are two implementations:  
  - `run_pd_defection_experiment.train_pd_repreader`: per-layer PCA on `defect - cooperate` differences.  
  - `pd_vector_extractor.compute_vectors_and_accuracy`: per-layer diff-of-means baseline.

## Metrics

- **Per-layer validation accuracy (`layer_accuracies`)**  
  For each layer, how often a held-out defection prompt projects more strongly in the learned direction than its matched cooperation prompt. This is computed on PD test pairs.

- **Best layer / best accuracy**  
  The layer index with the highest validation accuracy, and the corresponding value. Saved to `result.json` per PD run.

- **Decision rate (`_decision_rate`)**  
  On PD test prompts, how often the model's final logits prefer the defection option's label (`"A"` or `"B"`) over the cooperation label in a direct logit comparison.

- **Defection ratio (`_compute_defect_ratio`)**  
  On `GameTheoryDataset` PD instances, the fraction of valid responses where the parsed choice ID equals `2`, which is defined as the defection option for Prisoner's Dilemma in the benchmark.

## Steering and Hooks

- **Control layer(s)**  
  Transformer layers where the defection vector is added. Can be:  
  - A single layer (for example, the best layer from PD training).  
  - A block of layers (for example, the "middle third" of layers).

- `_register_control_hook`  
  Helper that installs a **forward hook** on a layer. At each forward pass, it:  
  - Broadcasts the steering vector to shape `(1, 1, hidden_dim)`.  
  - Adds it to the layer's hidden states (or the first element of the tuple if the layer returns a tuple).

- **Intensity**  
  Scalar multiplier on the steering vector before it is added in the hook. Higher intensity means a stronger push toward defection in the hidden space (but also higher risk of destabilizing the model).

## Benchmarking and Datasets

- `BenchmarkConfig` (`emotion_experiment_engine.data_models.BenchmarkConfig`)  
  Configuration object for benchmarks like `game_theory`. In this package it is constructed from `config/pd_behavior_game_theory.yaml`.

- `GameTheoryDataset` (`emotion_experiment_engine.datasets.games.GameTheoryDataset`)  
  Dataset wrapper for game-theory tasks. In this PD sub-package it is used to:  
  - Build PD prompts from `BenchmarkConfig` using a `GameBenchmarkPromptWrapper`.  
  - Serve items as dictionaries with a `"prompt"` key for the model.  
  - Provide static helpers to parse option choices (`_extract_options_from_prompt`, `_extract_option_from_response`).

- `GameBenchmarkPromptWrapper`  
  Builds prompts for game-theory tasks given a `PromptFormat` and a task type (for example, `Prisoners_Dilemma`).

- `PromptFormat` (`neuro_manipulation.prompt_formats.PromptFormat`)  
  Small helper that constructs full chat prompts from system/user/assistant messages, wrapping the tokenizer's chat formatting.

## Files and Artifacts

- `results/<model>_<timestamp>/result.json`  
  Summary of a PD defection training run: best layer, per-layer accuracies, defection rates vs intensity, etc.

- `results/<model>_<timestamp>/best_vector.npy`  
  NPZ file containing the selected per-layer defection vector.

- `results/layer_vectors/<model>/layer_<L>.npy`  
  Per-layer defection vectors for all layers considered in a run.

- `results/pd_behavior/<model>_pd_behavior_<timestamp>.json`  
  Summary of a PD behavior transfer run: defection ratios vs intensity on the game-theory benchmark.

- `results/delta/*`  
  Files produced by `compute_pd_delta`: `baseline.npz`, `steered.npz`, and `delta.npz` per run.
