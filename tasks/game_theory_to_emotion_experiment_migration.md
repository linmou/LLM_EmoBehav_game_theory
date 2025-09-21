## Game Theory → EmotionExperiment Migration Plan (TDD)

Purpose
- Unify game theory experiments under `EmotionExperiment` so all emotion experiments run via `python -m emotion_experiment_engine.memory_experiment_series_runner --config`.
- Keep the `emotion_experiment_engine` config design unchanged.
- Start with Prisoners_Dilemma only; add others later.
- For games, `evaluate_response` returns the chosen option number (1-based), not correctness; provide split evaluation (ratio of options).

Constraints
- No config shape changes in `emotion_experiment_engine`.
- No broad refactors; minimal adapters over invasive changes.
- Follow strict TDD (Red–Green–Refactor) with regression testing after each change.

Scope Summary (minimal surface area)
- Add a thin prompt-wrapper adapter for games (maps memory-style call signature to `GameReactPromptWrapper`).
- Add a dataset adapter that implements `BaseBenchmarkDataset` over existing game data.
- Register a single benchmark entry: `(name="game_theory", task_type="Prisoners_Dilemma")`.
- Add split evaluation aggregation (per-option ratios) to experiment outputs.

## Status
- [x] Red tests covering registry, prompt wrapper, dataset evaluation, ratio aggregation, and series runner dry-run.
- [x] Green implementation of game adapter, dataset, and benchmark registry entry.
- [x] Split choice ratio aggregation wired into result saving.
- [x] Regression pytest and mypy invoked (blocked by pre-existing repo issues; see run logs).
- [x] Sanity check runner executed via `python -m emotion_experiment_engine.memory_experiment_series_runner --config config/game_theory_prisoners_dilemma.yaml`.

Deliverables
- New benchmark registry entry for games.
- `GameTheoryDataset` (evaluation returns option number). 
- Prompt wrapper adapter for games.
- New CSVs including split ratios.
- Example YAML using `name: game_theory`.

Test-Driven Development Plan

1) Red: Unit + Integration tests (no production code yet)
- test_game_registry_integration
  - Given: PromptFormat (dummy), `create_benchmark_components("game_theory", "Prisoners_Dilemma", ...)`
  - Expect: returns callable prompt wrapper partial, answer wrapper partial, dataset instance.
  - Dataset `__getitem__` returns dict with keys {item, prompt, ground_truth}; and `collate_fn` returns {prompts, items, ground_truths}.

- test_game_prompt_wrapper_adapter_signature
  - Given: adapter wrapper instance with task_type=Prisoners_Dilemma
  - When: call with `(context=None, question=some_event, user_messages, enable_thinking, augmentation_config, answer, emotion, options=[...])`
  - Expect: returns a non-empty prompt string; internally uses `GameReactPromptWrapper` semantics (contains options).

- test_game_evaluate_response_regex_only
  - Given: prompt text that includes multiple-choice options; a generated output containing JSON-like `"decision": "..."` field that matches one option (case-insensitive containment).
  - Expect: `GameTheoryDataset.evaluate_response(...)` returns the 1-based option number as float.

- test_game_evaluate_response_llm_fallback
  - Mock: `emotion_experiment_engine.evaluation_utils.oai_response` (or corresponding utility) to return a dict with `option_id` when regex fail.
  - Expect: `evaluate_response` returns the mocked option number when regex extraction fails.

- test_series_runner_games_dry_run
  - Given: Minimal YAML config with one model and one benchmark `{name: game_theory, task_type: Prisoners_Dilemma}`.
  - Run: `MemoryExperimentSeriesRunner(..., dry_run=True)`
  - Expect: completes without errors; logs indicate dataset size > 0.

- test_split_evaluation_ratio
  - Given: synthetic results data with `score` values as option numbers (1..N), across (emotion,intensity,repeat_id).
  - When: call the aggregation function (added in experiment layer) to compute ratios per option.
  - Expect: returns a table with per-option proportions summing to 1 per (emotion,intensity[,repeat_id]).

2) Green: Minimal production code to pass tests
- Add `GameBenchmarkPromptWrapper` adapter class:
  - Constructor: `(prompt_format, task_type)` → resolve decision_class via `games/game_configs.get_game_config(task_type)`; wrap `GameReactPromptWrapper`.
  - `__call__(context, question, user_messages, enable_thinking, augmentation_config, answer, emotion, options)` → ignore unused args; build prompt with `event=question`, `options=options`.

- Add `GameTheoryDataset` (extends `BaseBenchmarkDataset`):
  - `_load_and_parse_data`: use `get_game_config(task_type)` to load scenarios; build `BenchmarkItem` with:
    - id: sequential index or source id
    - input_text: scenario event text
    - context: None
    - ground_truth: None
    - metadata: include list of options
  - `evaluate_response(response, ground_truth, task_name, prompt)`: parse options from `prompt`; extract choice:
    - Try regex on `response` to find `decision`
    - Map decision → first matching option (case-insensitive containment)
    - If regex fails, call LLM fallback (GPT-4o-mini) to obtain option_id
    - Return float(option_id)
  - `get_task_metrics`: return ["option_id"].

- Register benchmark spec in `benchmark_component_registry`:
  - Key: ("game_theory", "*") → dataset_class=GameTheoryDataset, answer_wrapper=IdentityAnswerWrapper, prompt_wrapper=GameBenchmarkPromptWrapper

- Add split evaluation aggregation in `EmotionExperiment` results step:
  - Produce `summary_choice_ratio.csv` with per-option ratios grouped by (emotion, intensity, repeat_id) and an overall by (emotion, intensity).

3) Refactor: Clean, concise, no overdesign
- Extract shared regex decision parsing (used by games) into a tiny utility to avoid duplication; keep it close to dataset.
- Keep adapter/dataset isolated; avoid changing runner or base dataset interfaces.
- Ensure no optional “configuration knobs” are added—smart defaults only.

4) Regression (回归测试)
- Run full suite (memory + new game tests) after each Green/Refactor step.
- mypy type-check modified files.

Acceptance Criteria
- Able to run: `python -m emotion_experiment_engine.memory_experiment_series_runner --config <yaml>` where yaml includes `{name: game_theory, task_type: Prisoners_Dilemma}`.
- Dry-run works; real run generates outputs under the same result structure as memory tasks.
- For games, `score` column holds the chosen option number (float), not correctness.
- `summary_choice_ratio.csv` exists with per-option ratios; ratios per group sum to 1.
- No changes to config schema in `emotion_experiment_engine`.
- Existing memory benchmark tests continue to pass.

Planned File Changes (minimal)
- Add: `emotion_experiment_engine/game_prompt_wrapper.py` (adapter)
- Add: `emotion_experiment_engine/datasets/games.py` (dataset)
- Update: `emotion_experiment_engine/benchmark_component_registry.py` (registry entry only)
- Update: `emotion_experiment_engine/experiment.py` (add split-ratio aggregation and write `summary_choice_ratio.csv`)
- Add tests:
  - `emotion_experiment_engine/tests/integration/test_game_registry_integration.py`
  - `emotion_experiment_engine/tests/unit/test_game_prompt_wrapper_adapter.py`
  - `emotion_experiment_engine/tests/unit/test_game_evaluation_extraction.py`
  - `emotion_experiment_engine/tests/e2e/test_series_runner_games_dry_run.py`
  - `emotion_experiment_engine/tests/unit/test_split_choice_ratio.py`

Rollout Notes
- Phase 1 supports Prisoners_Dilemma only; subsequent PRs can add other games by reusing the same adapter/dataset with different `task_type`.
- If desired later, we can add a convenience CSV (`choice_distribution.csv`) pivoted with columns `option_1,...,option_k` for quicker plotting.

Risks & Mitigations
- Option parsing from prompt: keep a robust, tested options parser that tolerates formatting.
- LLM fallback latency: used only when regex fails; batch size is small for post-processing.
- Maintaining config simplicity: single benchmark name `game_theory` avoids proliferating entries and keeps runners unchanged.
