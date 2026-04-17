# Annotate Stimulus

This module provides functionality to annotate stimuli with their corresponding trigger types using the OpenAI API. The `annotate_stimulus` function has been optimized to run in parallel using Python's `multiprocessing` module, allowing for faster processing of large datasets.

It also contains transformation scripts that convert structured research artifacts into
loadable scenario datasets for downstream game classes.

## Functionality

- **annotate_stimulus**: Annotates a list of stimuli with their trigger types based on a specified emotion. Utilizes multiprocessing to enhance performance.

## Usage

1. Ensure the OpenAI API key and base URL are set in the environment variables.
2. Prepare the input data in JSON format.
3. Call the `annotate_stimulus` function with the appropriate parameters.

### Social Game Transform CLI

The `transform_social_game_cases.py` CLI converts curated social game JSONL cases into a
success-only dataset that can be loaded by the corresponding game class while writing
failures, skips, and run metadata as separate machine-readable artifacts.

Supported first-release social games:

- `beauty_contest` -> validates through `BeautyContestScenario`
- `escalation_game` -> validates through `EscalationGameScenario`

For `escalation_game`, the scenario contract now accepts optional explicit
`previous_actions` while still allowing fallback `previous_actions_length`. When both
are present, the length must match the explicit history or validation fails.

Example:

```bash
python -m data_creation.transform_social_game_cases \
  --social-game beauty_contest \
  --input-path /home/jjl7137/diplomacy_cicero/social_game_outputs/beauty_contest/curated_cases/beauty_contest_cases.jsonl \
  --few-shot-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/beauty_contest_few_shot_examples.json \
  --rubric-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/transform_rubrics.md \
  --output-dir /tmp/beauty_contest_transform_run
```

Escalation example:

```bash
python -m data_creation.transform_social_game_cases \
  --social-game escalation_game \
  --input-path /home/jjl7137/diplomacy_cicero/social_game_outputs/escalation_game/curated_cases/escalation_game_cases.jsonl \
  --few-shot-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/beauty_contest_few_shot_examples.json \
  --rubric-path /home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/transform_rubrics.md \
  --output-dir /tmp/escalation_game_transform_run
```

Artifacts:

- `beauty_contest.success.json`: loadable transformed rows only
- `beauty_contest.failures.jsonl`: invalid or unsuccessful rows
- `beauty_contest.skipped.jsonl`: resumed rows skipped because they were already finalized
- `escalation_game.success.json`: loadable transformed rows only
- `escalation_game.failures.jsonl`: invalid or unsuccessful rows
- `escalation_game.skipped.jsonl`: resumed rows skipped because they were already finalized
- `run_metadata.json`: counts, input/output paths, and completed identities
