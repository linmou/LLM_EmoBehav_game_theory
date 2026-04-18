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

# Transform Social game cases

## supportive docs for .agents/skills/diplomacy-social-game-transform

 data_creation/ransform_to_natural_lannguage_samples storages rubrics and fewshot examples 

### Social Game Transform CLI

The `transform_social_game_cases.py` CLI converts curated social game JSONL cases into a
success-only dataset that can be loaded by the corresponding game class while writing
failures, skips, and run metadata as separate machine-readable artifacts.

Supported first-release social games (will support all social games):

- `beauty_contest` -> validates through `BeautyContestScenario`
- `escalation_game` -> validates through `EscalationGameScenario`
- `trust` -> validates through `TrustGameTrusteeScenario`

For `escalation_game`, the scenario contract now accepts optional explicit
`previous_actions` while still allowing fallback `previous_actions_length`. When both
are present, the length must match the explicit history or validation fails.

Few-shot selection is now constrained by the selected same-game asset file:

- the first example pool is the full selected `--few-shot-path` file
- the runtime pool is filtered to variants present in the current input run
- each row must keep at least one same-variant example in its prompt pack
- multi-variant runs fail with an explicit `few_shot_selection` failure record when the filtered pool cannot supply the required cross-variant examples for a row
- `run_metadata.json` records the run-present variant set so the filtered few-shot pool can be reconstructed later

Example:

```bash
python -m data_creation.transform_social_game_cases \
  --social-game beauty_contest \
  --input-path /home/jjl7137/diplomacy_cicero/social_game_outputs/beauty_contest/curated_cases/beauty_contest_cases.jsonl \
  --output-dir /tmp/beauty_contest_transform_run
```

Escalation example:

```bash
python -m data_creation.transform_social_game_cases \
  --social-game escalation_game \
  --input-path /home/jjl7137/diplomacy_cicero/social_game_outputs/escalation_game/curated_cases/escalation_game_cases.jsonl \
  --output-dir /tmp/escalation_game_transform_run
```

Default prompt assets now live under:

- `data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md`
- `data_creation/transform_to_natural_lannguage_samples/diplomacy/<game_name>_few_shot_examples.json`

Override `--few-shot-path` or `--rubric-path` only when you intentionally want non-default prompt assets.

Artifacts:

- `escalation_game.success.json`: loadable transformed rows only
- `escalation_game.failures.jsonl`: invalid or unsuccessful rows
- `escalation_game.skipped.jsonl`: resumed rows skipped because they were already finalized

- `run_metadata.json`: counts, input/output paths, and completed identities
