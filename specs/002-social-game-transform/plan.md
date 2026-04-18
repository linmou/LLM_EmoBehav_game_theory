# Implementation Plan: Social Game Case Transformation Pipeline

**Branch**: `002-social-game-transform` | **Date**: 2026-04-17 | **Spec**: `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/spec.md`
**Input**: Feature specification from `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/spec.md`

## Summary

Refine the existing `data_creation.transform_social_game_cases` CLI so few-shot selection becomes explicit, per-row, and diversity-driven without changing the core success-only artifact contract. The implementation will keep real scenario-class validation, resume bookkeeping, visible progress reporting, and explicit failure artifacts, while replacing the old single-pack assumption with per-row few-shot packs built from same-social-game, run-present variants only, scored on `description` plus `behavior_choices` using a greedy weighted 3/4/5-gram gain rule.

## Research-Impacting Factors

- **Prompt design**: Few-shot selection changes from one static few-shot file load to per-row pack construction with exact composition rules.
- **Data format**: Run metadata and supporting artifacts must preserve enough information to audit few-shot eligibility, selected examples, and diversity reports.
- **Evaluation method**: Validation still uses `scenario_class(**data)`, while few-shot quality is judged with classic n-gram diversity metrics rather than semantic similarity heuristics, and CLI output must expose visible progress plus a final summary.

## Technical Context

**Language/Version**: Python 3.11.x in the repository Python environment  
**Primary Dependencies**: Standard library (`argparse`, `json`, `pathlib`, `concurrent.futures`, `datetime`, `re`), `openai`, `python-dotenv`, in-repo game contracts from `games.game_configs` and scenario classes under `games/`  
**Storage**: Local Markdown, JSON, and JSONL files for prompt assets, success artifacts, failure/skipped artifacts, candidate artifacts, diversity reports, and run metadata  
**Testing**: `pytest` for CLI and artifact tests in `data_creation/tests/test_transform_social_game_cases.py`; `mypy` on modified Python modules during implementation  
**Target Platform**: Local research CLI runs in the repository environment on workstation or batch shells  
**Project Type**: Internal data-processing CLI pipeline  
**Performance Goals**: Support long-running transformation jobs over at least 100-600 source rows with configurable worker concurrency, visible progress, resumability, and bounded JSON/JSONL artifact writes  
**Constraints**: No silent fallbacks; validate successful outputs with the real scenario constructor; fail loudly when few-shot pools cannot satisfy the per-row composition rule; preserve explicit provenance and resume identity; keep the design simple and file-based  
**Scale/Scope**: First release supports exactly `beauty_contest` and plain `escalation_game`; each source row builds its own few-shot pack from one same-game example library filtered by variants present in the current run input

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Config-truth and no silent fallbacks**: PASS. The plan keeps explicit social-game mapping, explicit variant filtering, and explicit failure when same-variant or cross-variant pools are insufficient.
- **Real downstream contract validation**: PASS. Successful rows still cross the real `scenario_class(**data)` boundary before entering the success dataset.
- **Test-first and regression discipline**: PASS. Implementation work must start with failing tests for per-row few-shot pack selection, artifact bookkeeping, and unsupported-pool failures, followed by regression pytest and mypy on modified Python code.
- **Progress, resumability, provenance, explicit failure accounting**: PASS. Existing run accounting remains in scope, and the design extends metadata and supporting artifacts rather than weakening them.
- **Simplicity**: PASS. The design modifies one existing CLI and its tests instead of adding a service, registry framework, or semantic-selection subsystem.

## Project Structure

### Documentation (this feature)

```text
specs/002-social-game-transform/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── social-game-transform-cli.md
└── tasks.md
```

### Source Code (repository root)

```text
data_creation/
├── transform_social_game_cases.py
├── tests/
│   └── test_transform_social_game_cases.py
└── README.md

games/
├── escalation_game.py
├── beauty_contest.py
└── game_configs.py

transform_rubrics.md
beauty_contest_few_shot_examples.json
escalation_game_few_shot_examples.json
```

**Structure Decision**: Keep the feature inside the existing `data_creation` CLI and test module. Extend the prompt-asset and selection logic in place, keep game contracts in `games/`, and document the user-facing CLI behavior in the feature contracts folder.

## Phase 0: Research Outcome

Phase 0 resolved the design unknowns around few-shot selection:

- Few-shot eligibility is limited to the same social game and variants present in the current run input.
- The first pool is the full contents of the selected same-game `few-shot` file, before any run-level filtering.
- Diversity scoring uses only `description` and `behavior_choices`.
- Ranking uses a deterministic greedy weighted 3/4/5-gram gain rule.
- Selection is per source row, with exactly 2 cross-variant examples and all remaining examples from the row's own variant.
- Pack construction fails loudly if the eligible pool cannot satisfy that composition rule.

See `/Users/admin/Documents/GitHub.nosynchr/LLM_EmoBehav_game_theory_social_game_transform/specs/002-social-game-transform/research.md`.

## Phase 1: Design Plan

### Data Model Changes

- Extend prompt-pack modeling to distinguish:
  - base file-loaded example pools
  - run-level prompt assets
  - run-matched eligible pools
  - per-row few-shot packs
  - candidate/diversity audit artifacts
- Add explicit validation rules for:
  - missing run-present variant coverage
  - insufficient same-variant pool
  - insufficient cross-variant pool
  - exact `2` cross-variant example requirement
- Preserve the existing success/failure/skipped/run-metadata model and real scenario-constructor validation.

### Contract Changes

- Update the CLI contract so `--few-shot-path` is treated as the same-game few-shot example file rather than a static pack.
- Document the per-row selection rules, candidate artifact, diversity report behavior, and visible progress or final summary expectations.
- Keep unsupported games, invalid identities, and invalid pools as fatal setup or explicit failure cases rather than relaxed behavior.

### Quickstart Changes

- Show `beauty_contest` and `escalation_game` commands using their own few-shot example files.
- Include candidate-generation flags and verification steps for diversity outputs.
- Document how to confirm per-row selection composition, visible progress output, and contract-valid success artifacts.

### Implementation Slices

1. Add failing tests for unsupported-game rejection, invalid identity handling, and per-row few-shot eligibility with exact cross-variant composition.
2. Keep User Story 1 focused on loadable output and explicit failure accounting, while User Story 2 owns the exact same-game pool derivation and per-row pack-construction rules.
3. Refactor prompt loading so the CLI loads the full same-game few-shot example file, indexes it by variant, and derives run-present variants from the selected input rows.
4. Build per-row few-shot packs using deterministic greedy weighted n-gram gain scoring over `description` and `behavior_choices`, and add explicit narrative-consistency/style validation work.
5. Extend CLI output, supporting artifacts, and metadata so runs expose visible progress, final summaries, candidate generation, selected packs, and diversity metrics without weakening the success-only dataset contract.
6. Run regression pytest and mypy on modified Python files.

## Post-Design Constitution Check

- **Config-truth and no silent fallbacks**: PASS. The design removes the stale temporary cross-game few-shot reuse rule and replaces it with explicit, testable selection constraints.
- **Real downstream contract validation**: PASS. No change weakens constructor validation or success-only artifact boundaries.
- **Test-first and regression discipline**: PASS. The plan requires new failing tests before implementation and preserves regression plus mypy obligations.
- **Reproducible pipelines**: PASS. The design keeps progress/resume artifacts and adds more selection auditability rather than less.
- **Simplicity**: PASS. The plan stays inside one CLI module and one test module, with no new infrastructure layers.

## Complexity Tracking

No constitution exceptions are required for this plan.
