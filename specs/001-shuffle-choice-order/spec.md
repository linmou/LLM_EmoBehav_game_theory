# Feature Specification: Shuffle Behavior Option Order and Dual Choice Ratios

**Feature Branch**: `001-shuffle-choice-order`
**Created**: 2025-12-07
**Status**: Draft
**Input**: User description: "dont build new branch, just use this branch. new feature: shuffle behavior options. for each dataset , there are some options indicating different decision. please see behavior_choices of each data sample. Now please when build GameTheoryDataset (emotion_experiment_engine/datasets/games.py), shuffle the choices so that each data sample has different options order, but still use option 1/2 as index . but record the both choosed option id and the relavant indication (e.g. defect or cooperate) in the sample. When finally calculate the choice ratio, repot both the option indication ratio and id ratio."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Separate behavior preference from position bias (Priority: P1)

Researchers run game-theoretic experiments where models choose between behavior options (for example, "cooperate" vs "defect"). They want the order of options shown in each scenario to vary so that any bias toward "option 1" or "option 2" does not confound whether the model prefers a particular behavior.

**Why this priority**: Without randomized option order, it is impossible to tell whether differences in choice rates are driven by behavior semantics or by a simple preference for a particular option index. This directly affects the validity of emotion and game-theory findings.

**Independent Test**: Run an experiment on a small dataset twice with different random seeds. Verify that for at least some scenarios the behavior labels swap positions between option indices while the recorded mapping from choice index to behavior label remains correct in the stored results.

**Acceptance Scenarios**:

1. **Given** a dataset of game scenarios with multiple behavior options, **When** researchers build the dataset for an experiment, **Then** the order of behavior options attached to each scenario is randomized while preserving a clear mapping between each option index and its underlying behavior.
2. **Given** a specific scenario in the dataset, **When** the experiment records the model's chosen option index, **Then** the stored data for that scenario also includes the corresponding behavior label (for example, "cooperate" or "defect") attached to that index.

---

### User Story 2 - Report both behavior-level and index-level choice ratios (Priority: P2)

Researchers analyze experiment logs to understand how emotions and prompts change decision patterns. They want summary tables that show both how often each behavior label is chosen and how often each option index (1, 2, 3, ...) is chosen, so they can quantify both semantic preference and position bias.

**Why this priority**: Existing analyses typically report a single "choice ratio" that blends behavior semantics with option position. To interpret emotional effects correctly, researchers must be able to see both behavior-level and index-level statistics.

**Independent Test**: Run a controlled experiment on a toy dataset where the ground-truth counts of each behavior label and each option index are known. Verify that the produced summaries match these counts for both behavior-level and index-level ratios.

**Acceptance Scenarios**:

1. **Given** an experiment run that records choices for many scenarios, **When** researchers generate choice-ratio summaries, **Then** the output includes (a) ratios by behavior label (for example, percent of "defect" vs "cooperate") and (b) ratios by option index (for example, percent of choices for option 1 vs option 2).
2. **Given** the same set of logged decisions, **When** researchers recompute ratios manually from the raw records, **Then** their results match the system's behavior-level and index-level ratios within rounding error.

---

### User Story 3 - Inspect individual samples for audit and debugging (Priority: P3)

Researchers sometimes need to audit individual decisions when something looks surprising in aggregate statistics. They want to inspect a single sample and see both the options presented (with their order), the chosen option index, and the corresponding behavior label for that choice.

**Why this priority**: Being able to inspect individual samples makes it easier to debug experiment setups, confirm that randomization is working, and explain surprising aggregate patterns.

**Independent Test**: Select a handful of logged decisions and inspect their stored representation. Verify that each contains the full list of behavior options in the order shown to the model, the chosen option index, and the chosen behavior label.

**Acceptance Scenarios**:

1. **Given** a stored decision record for a scenario, **When** a researcher inspects it, **Then** they can see the ordered list of behavior options, the option index chosen by the model, and the behavior label corresponding to that index.
2. **Given** any decision record included in the aggregate statistics, **When** a researcher recomputes that record's contribution to behavior-level and index-level ratios, **Then** it is clear how that record contributes to each summary.

---

### Edge Cases

- What happens when a scenario has more than two behavior options? The system must still randomize the order of all options and correctly track both the chosen index and behavior label for any of them.
- How does the system handle scenarios where behavior options are missing or cannot be mapped to clear behavior labels? Such scenarios must be excluded from ratio calculations or surfaced as explicit data-quality errors to researchers.
- How does the system handle individual decisions whose chosen option index does not exist in the scenario’s option list? Those decisions must still contribute to index-level ratios under their chosen option index and must appear in behavior-level ratios under an explicit `"unknown"` behavior bucket so that mapping failures are visible without breaking aggregation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST randomize the order of behavior options for each game-theoretic scenario when constructing the dataset used for experiments, such that option indices (1, 2, 3, ...) reflect the randomized order rather than a fixed canonical order.
- **FR-002**: The system MUST preserve a clear mapping between each option index and its underlying behavior label for every scenario, so that the semantic meaning of "option 1", "option 2", etc., is always recoverable from stored data.
- **FR-003**: For every recorded decision event, the system MUST store both (a) the chosen option index and (b) the corresponding behavior label derived from the randomized option list.
- **FR-004**: The choice-ratio reporting functionality MUST compute and expose behavior-level ratios (for example, fraction of "defect" vs "cooperate") based on the stored behavior labels, regardless of which option index those behaviors were assigned in each scenario.
- **FR-005**: The choice-ratio reporting functionality MUST compute and expose index-level ratios (for example, fraction of choices for option 1 vs option 2) based on the stored option indices, regardless of which behaviors those indices represent in each scenario.
- **FR-006**: For any experiment where both behavior-level and index-level ratios are computed, the system MUST ensure that the total number of decisions counted in both sets of ratios is identical, so that comparisons between behavior preference and position bias are statistically valid.
- **FR-007**: If a scenario lacks a valid list of behavior options, the system MUST either exclude that scenario from ratio calculations or surface a clear error to the researcher before analysis proceeds. For individual decisions where the chosen option index does not match any option in the scenario’s option list, the system MUST (a) include that decision in index-level ratios using the original option index and (b) assign it to an explicit `"unknown"` behavior bucket in behavior-level ratios so data-quality issues are visible without violating FR-006.

### Key Entities *(include if feature involves data)*

- **Behavior Option**: A labeled action available to the model in a game-theoretic scenario (for example, "cooperate", "defect", "trust", "betray"). Each behavior option may appear at different option indices across different scenarios due to randomization; this is the behavior *category* stored in item metadata.
- **Decision Record**: The stored representation of a single model decision in a scenario, including the full ordered list of behavior options shown (with their categories) and the chosen option index. The behavior category of a decision is derived by combining the chosen index with the per-sample Behavior Options.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a test experiment where the same dataset is run twice with different random seeds, at least one scenario per game type shows behavior labels appearing at different option indices across runs while preserving correct index-to-label mappings in the stored records.
- **SC-002**: For a synthetic dataset with known counts of chosen behaviors and chosen option indices, the system's behavior-level and index-level choice ratios match the expected counts (up to rounding) when recomputed from the stored decision records.
- **SC-003**: For any experiment with at least 100 decisions per behavior label, no single behavior label appears at the same option index in more than 70% of scenarios, indicating that option order randomization is effectively distributing behaviors across positions.
- **SC-004**: Researchers can generate a standard choice-ratio summary for an experiment and, without writing additional analysis code, view both behavior-level and index-level ratios side by side for comparison.

## Assumptions & Dependencies

- Existing experiment runners already log individual decisions in a structured way that can be extended to include both chosen option index and behavior label without redesigning the entire logging format.
- The analysis pipeline that currently computes choice ratios can be extended to consume the enriched decision records and produce separate behavior-level and index-level summaries.
- Experiments use a controllable source of randomness (such as a global random seed) so that option-order randomization can be made reproducible for debugging and comparison across runs.
