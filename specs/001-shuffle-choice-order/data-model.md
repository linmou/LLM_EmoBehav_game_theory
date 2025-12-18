# Data Model: Shuffle Behavior Option Order and Dual Choice Ratios

**Feature**: 001-shuffle-choice-order
**Spec**: `specs/001-shuffle-choice-order/spec.md`
**Plan**: `specs/001-shuffle-choice-order/plan.md`

---

## Entities

### 1. BehaviorOption

- **Description**: A labeled action available to the model in a game-theoretic scenario.
- **Fields**:
  - `option_index` (int): Position of the option in the randomized list for a given sample (1-based).
  - `behavior_label` (str): Semantic label such as `"cooperate"`, `"defect"`, `"trust"`, `"betray"`.
- **Identity & uniqueness**:
  - Within a single sample, `option_index` MUST be unique.
  - The same `behavior_label` MAY appear at different `option_index` values across samples due to shuffling.
  - In persisted metadata and CSV summaries, this label is serialized under the field name `behavior` (for example, `options[i]["behavior"]` and the `behavior` column in `summary_behavior_ratio.csv`).

### 2. GameSample

- **Description**: Single row/sample in `GameTheoryDataset` after shuffling behavior options.
- **Fields**:
  - `sample_id` (str or int): Stable identifier (e.g., dataset index or scenario id).
  - `behavior_options` (list[BehaviorOption]): Randomized list of behavior options presented to the model.
  - `metadata` (dict): Existing game metadata (game type, emotion, payoff info, etc.; unchanged by this feature).
- **Identity & uniqueness**:
  - `sample_id` uniquely identifies a sample within a dataset.
  - For each `GameSample`, `behavior_options` MUST contain the same behaviors as the original configuration, but in randomized order.

### 3. DecisionRecord

- **Description**: Representation of a single model decision for a `GameSample`. In implementation this aligns with `ResultRecord`, where `score` encodes the chosen option index and the behavior category is looked up from the corresponding `BehaviorOption` for that sample.
- **Fields**:
  - `sample_id` (str or int): Links back to the corresponding `GameSample`.
  - `chosen_option_index` (int): The index (1-based) of the chosen behavior option.
  - `emotion` (str): Emotion condition under which the decision was made (e.g., `"anger"`, `"happiness"`).
  - `model_name` (str): Identifier for the model used (e.g., `"Qwen2.5-0.5B-anger"`).
- **Identity & uniqueness**:
  - (`sample_id`, `emotion`, `model_name`) SHOULD uniquely identify a single `DecisionRecord` within a run.

### 4. ChoiceRatioSummary

- **Description**: Aggregated statistics over many `DecisionRecord` instances, using behavior categories from `BehaviorOption` and chosen indices from decisions.
- **Fields**:
  - `behavior_choice_counts` (dict[str, int]): Count of decisions per behavior label (including an `"unknown"` bucket when a chosen option_id does not match any option in the scenario).
  - `index_choice_counts` (dict[int, int]): Count of decisions per chosen option index.
  - `behavior_choice_ratios` (dict[str, float]): Normalized ratios per behavior label (counts / total decisions); labels correspond to the `behavior` field in metadata and CSV outputs.
  - `index_choice_ratios` (dict[int, float]): Normalized ratios per option index (counts / total decisions).
  - `total_decisions` (int): Total number of `DecisionRecord` instances included.
- **Identity & uniqueness**:
  - Summary MAY be computed per (emotion, model_name, game type) or over the entire dataset; the feature does not constrain grouping strategy.

---

## Relationships

- Each `GameSample` has a list of `BehaviorOption` entries; this list is randomized per sample.
- Each `DecisionRecord` references exactly one `GameSample` via `sample_id`.
- `ChoiceRatioSummary` aggregates over a collection of `DecisionRecord` entries and does not directly reference individual samples.

---

## Validation Rules

- For every `GameSample`, `behavior_options` MUST contain at least two `BehaviorOption` entries.
- For every `GameSample`, `behavior_options.option_index` MUST form a contiguous sequence starting at 1.
- For every `DecisionRecord` where the chosen option_id matches a `BehaviorOption`, the effective behavior label MUST equal that option’s `behavior_label` (serialized as `behavior`). If no matching `BehaviorOption` exists for the chosen option_id, the effective behavior label MUST be recorded as `"unknown"` in behavior-level summaries.
- For any `ChoiceRatioSummary`, the sum of `behavior_choice_counts` and the sum of `index_choice_counts` MUST both equal `total_decisions`.
