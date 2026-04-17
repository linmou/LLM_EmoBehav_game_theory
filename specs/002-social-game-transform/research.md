# Research: Social Game Case Transformation Pipeline

## Decision 1: Keep the feature as a Python CLI under `data_creation/`

- Decision: Continue implementing the transformation workflow as a file-based Python CLI in `data_creation/`.
- Rationale: The repository already treats scenario and dataset generation as script-driven research work. This keeps the feature close to the existing tests, prompt assets, and artifact-writing patterns without inventing a service layer.
- Alternatives considered:
  - Web service: rejected because this is an offline preprocessing step, not an always-on runtime dependency.
  - Shell-only pipeline: rejected because prompt assembly, JSON normalization, and contract validation are cleaner and more testable in Python.

## Decision 2: Use explicit per-game mapping, not auto-discovery

- Decision: Represent supported social games with an explicit mapping that carries the target scenario class, canonical target game name, artifact filenames, prompt wording, and deterministic field injectors.
- Rationale: The current code is hardcoded for Beauty Contest. A small explicit mapping is the minimum change that makes dual-game support real while preserving loud rejection for unsupported games.
- Alternatives considered:
  - Auto-discover from `games/`: rejected because the transform contract also needs prompt and artifact behavior, not just a Python import.
  - Generic registry framework: rejected because it adds abstraction without a real third or fourth game requirement.

## Decision 3: Validate success rows with the real scenario constructor

- Decision: Keep `scenario_class(**data)` as the decisive validation rule for successful transformed rows, using the mapped class for each supported game.
- Rationale: The feature exists to produce rows that the real game loader can consume. The scenario class is the authoritative downstream contract and catches shape or semantic mismatches that a duplicate local schema would miss.
- Alternatives considered:
  - Standalone local schema only: rejected because it would drift from the actual consumer.
  - Loose shape checks plus later runtime validation: rejected because it delays the real failure until downstream experiments.

## Decision 4: Keep success-only output and separate machine-readable failure artifacts

- Decision: Preserve the current success dataset / failures JSONL / skipped JSONL / run metadata split.
- Rationale: Downstream loaders should never need to filter mixed-status rows. This repo’s constitution also prefers explicit accounting of broken rows over silent omission.
- Alternatives considered:
  - Mixed output with status flags: rejected because it weakens the loadable-contract boundary.
  - Abort on first bad row: rejected because long-running transformation jobs must continue over valid rows.

## Decision 5: Reuse the existing OpenAI-compatible DeepSeek client pattern

- Decision: Continue using the repository’s OpenAI-compatible client pattern with `DPSK_API` and optional `.env` loading.
- Rationale: The current implementation already works that way, and the feature scope is about contract correctness, not client abstraction.
- Alternatives considered:
  - Raw HTTP calls: rejected because they duplicate client behavior and error handling.
  - New LLM wrapper layer: rejected because it adds complexity without solving a current problem.

## Decision 6: First release supports exactly `beauty_contest` and plain `escalation_game`

- Decision: Expand first-release support to two social games: `beauty_contest` and plain `escalation_game`, while keeping `Diplomacy_Escalation_Game` out of scope.
- Rationale: This satisfies the current user goal while keeping the contract surface narrow enough to test properly.
- Alternatives considered:
  - Beauty Contest only: rejected because it leaves the current request unmet.
  - Include diplomacy escalation too: rejected because that is a distinct domain framing and would blur tests and prompt contracts.

## Decision 7: Temporarily reuse Beauty Contest few-shot examples for `escalation_game`

- Decision: In this release, allow `escalation_game` to reuse the existing `beauty_contest` few-shot asset while still applying the `Escalation_Game` runtime contract.
- Rationale: The spec explicitly allows this temporary shortcut, and it avoids blocking implementation on prompt asset creation. The real scenario constructor remains the guardrail.
- Alternatives considered:
  - Require dedicated escalation examples now: rejected by clarified scope for this release.
  - Run escalation with rubric only: rejected because it would make prompt behavior less anchored and harder to compare.

## Decision 8: Inject deterministic per-game structural fields from code

- Decision: Inject canonical fields such as `game_name`, `payoff_matrix`, scenario class selection, and artifact filenames from code/config rather than expecting the model to invent them.
- Rationale: These fields are fixed by the repository’s own game contracts and should not be left to model output variance.
- Alternatives considered:
  - Let the model generate all fields: rejected because deterministic contract fields are not a language task.
  - Inject only `payoff_matrix`: rejected because canonical naming and contract mapping should come from the same explicit source.

## Decision 9: For `Escalation_Game`, explicit history is primary but optional

- Decision: Extend `EscalationGameScenario` so `previous_actions` is optional but primary when present, while `previous_actions_length` remains a fallback representation for simpler/generated cases.
- Rationale: Explicit history is semantically richer and better for transformed real cases, but requiring it everywhere would unnecessarily break simpler existing uses.
- Alternatives considered:
  - Keep only `previous_actions_length`: rejected because it loses meaningful history when source-derived actions exist.
  - Make `previous_actions` mandatory: rejected because the user explicitly does not want that.

## Decision 10: Reject contradictory dual-history inputs

- Decision: When an `Escalation_Game` row includes both `previous_actions` and `previous_actions_length`, require `previous_actions_length == len(previous_actions)` or fail validation.
- Rationale: Silent mismatch would hide bad data and undermine trust in the transformed scenarios.
- Alternatives considered:
  - Ignore `previous_actions_length` when history is present: simpler, but it hides contradictory input.
  - Let `previous_actions_length` rebuild history: rejected because it throws away the richer explicit data.
