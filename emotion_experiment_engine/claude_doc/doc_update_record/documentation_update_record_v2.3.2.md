# Documentation Update Record v2.3.2

Intent: record the documentation changes for the game-theory option shuffle seed semantics update.

Date: 2026-04-18

## Changes

- Updated `emotion_experiment_engine/README.md` to document `augmentation_config.shuffle_options_seed` for reproducible option shuffling in `game_theory` and `game_theory_decision` benchmarks.
- Updated `emotion_experiment_engine/claude_doc/data_flow_and_integration_points.md` to clarify that `augmentation_config` is merged into the game config before `GameTheoryDataset` builds options.
- Documented that `behavior_ratio` is analysis semantics and must not be used as an option-shuffle seed.
