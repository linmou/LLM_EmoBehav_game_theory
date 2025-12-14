# Documentation Update Record v2.2.3

- 2025-09-28: Added offline SWE-bench adapter and registry entry.
  - New file: `emotion_experiment_engine/datasets/swebench.py` (HF save_to_disk loader; passes through `text_inputs`).
  - Registry: mapped (`"swebench"`, `"patch"`) to `SWEbenchDataset` with no explicit prompt wrapper.
  - Tests: added smoke tests `emotion_experiment_engine/tests/test_swebench_dataset.py` and `emotion_experiment_engine/tests/test_registry_swebench.py`.
  - Predictions: extended `emotion_experiment_engine/experiment.py` to capture SWE-bench predictions JSONL and annotate run metadata; added `emotion_experiment_engine/tests/test_swebench_predictions.py`.
  - Config: created `config/swebench_series_lite.yaml` for deferred generation smoke runs.
  - Plan doc: updated `tasks/swebench_migration_plan.md` to include dry-run validation command and fixed typo in Phase 3 header.
  - Commit: N/A (applied via Codex CLI patch).
- **Date**: 2025-12-07
- **Files Updated**:
  - `emotion_experiment_engine/claude_doc/testing_strategy_and_coverage_analysis.md`
- **Summary of Changes**:
  - Documented game_theory choice-ratio testing updates, including the `"unknown"` behavior bucket for unmappable option ids.
  - Clarified key tests and CSV column naming (`behavior`, including `"unknown"`), and listed targeted pytest subsets for fast feedback.
