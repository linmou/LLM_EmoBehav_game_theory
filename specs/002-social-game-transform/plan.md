# Implementation Plan: Social Game Case Transformation Pipeline

**Branch**: `002-social-game-transform` | **Date**: 2026-04-17 | **Spec**: [/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/spec.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/spec.md)
**Input**: Feature specification from `/specs/002-social-game-transform/spec.md`

## Summary

Generalize the existing `data_creation.transform_social_game_cases` CLI from a Beauty-Contest-only implementation into an explicit two-game pipeline supporting `beauty_contest` and plain `escalation_game`. Keep the design simple: drive per-game behavior from explicit mapping, validate success rows with the real scenario constructors, preserve success-only output artifacts plus failure/skip metadata, and extend `EscalationGameScenario` so optional explicit `previous_actions` can coexist with the existing `previous_actions_length` fallback under strict consistency validation.

## Technical Context

**Language/Version**: Python 3.10+ in the repository conda environment  
**Primary Dependencies**: standard library (`argparse`, `json`, `pathlib`, `datetime`, `concurrent.futures`), `pydantic`, `openai`, `python-dotenv`, in-repo game modules under `games/`  
**Storage**: Local files only: JSONL source input, JSON success dataset, JSONL failure/skipped artifacts, JSON run metadata, Markdown prompt assets  
**Testing**: `pytest`, targeted game contract tests, CLI/integration tests, `mypy` for modified Python files  
**Target Platform**: Linux CLI environment in this repository workspace  
**Project Type**: Internal research CLI/data pipeline  
**Performance Goals**: Reliable processing of one curated JSONL dataset per run with visible per-row progress, resumability, and deterministic artifact accounting; throughput is secondary to correctness and contract validity  
**Constraints**: No silent fallbacks, no provider auto-switching, success dataset must remain success-only, explicit unsupported-game rejection, plain `Escalation_Game` only, keep shell scripting simple, keep code structure minimal  
**Scale/Scope**: One feature-focused CLI module plus aligned scenario-class updates and tests; expected dataset scale is tens to low-thousands of curated rows per run

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Config-truth / no silent fallbacks**: PASS. The plan keeps explicit social-game mapping, explicit prompt assets, and loud rejection for unsupported games. The one approved temporary exception, reusing the Beauty Contest few-shot asset for `escalation_game`, is declared in the spec and will be covered by tests.
- **Real downstream contract validation**: PASS. Success rows will continue to be validated with `scenario_class(**data)` using the actual mapped scenario class for each supported game.
- **Test-first + regression + mypy**: PASS. The implementation plan assumes failing tests first for the CLI and scenario contract changes, then targeted regression coverage and `mypy` over modified Python files.
- **Progress / resumability / provenance / explicit failure**: PASS. Existing artifact bookkeeping stays in place and is extended, not replaced.
- **Simplicity**: PASS. The plan uses one explicit mapping table, one normalization path for game-specific injected fields, and one scenario-class contract boundary. No service layer or generalized registry is needed.

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
├── README.md
└── tests/
    ├── features/
    │   └── social_game_transform.feature
    └── test_transform_social_game_cases.py

games/
├── beauty_contest.py
├── escalation_game.py
├── game_configs.py
└── payoff_matrices.py

tests/
└── games/
    └── test_beauty_contest_game_config.py
```

**Structure Decision**: Keep the feature in the existing `data_creation/` CLI pipeline and extend the real scenario contracts in `games/`. Add tests at the two true boundaries: pipeline behavior in `data_creation/tests/` and game-load validation in `tests/games/` plus any dedicated `games/escalation_game.py` regression coverage needed for history normalization.

## Phase 0: Research

Research resolved these design decisions:

1. Keep the feature as a Python CLI under `data_creation/`.
2. Continue using the repository’s OpenAI-compatible DeepSeek pattern.
3. Generalize via explicit per-game mapping, not auto-discovery.
4. Validate success rows with the real mapped scenario class.
5. Keep success-only outputs with separate failure/skipped artifacts.
6. Preserve `id + source.game_id` as the resume identity.
7. Treat `Escalation_Game` history as optional explicit `previous_actions` plus fallback `previous_actions_length`, with strict mismatch rejection when both are present.

Research output: [/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/research.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/research.md)

## Phase 1: Design & Contracts

### Data Model

Define the core entities for:

- source curated rows and identity bookkeeping
- prompt-pack mapping and asset usage
- per-game transform contracts for `beauty_contest` and `escalation_game`
- explicit `Escalation_Game` history representation and validation precedence
- success/failure/run-metadata artifacts

Design output: [/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/data-model.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/data-model.md)

### Interface Contract

Document the CLI contract for:

- supported values for `--social-game`
- required and optional arguments
- artifact names per supported game
- exit behavior and progress output
- validation rules for prompt assets, source identity, and scenario loading

Contract output: [/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/contracts/social-game-transform-cli.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/contracts/social-game-transform-cli.md)

### Quickstart

Document one runnable flow for each first-release supported game, including the temporary few-shot reuse rule for `escalation_game`, expected artifacts, and constructor-based verification.

Quickstart output: [/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/quickstart.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/quickstart.md)

### Implementation Shape

The planned code changes are:

- `data_creation/transform_social_game_cases.py`
  - replace Beauty-Contest-only mapping with explicit support for `beauty_contest` and `escalation_game`
  - map each social game to scenario class, canonical game name, artifact filenames, prompt wording, and deterministic injected fields
  - keep history fields source-derived/model-derived, but validate them against the target scenario contract
- `games/escalation_game.py`
  - extend the scenario contract to accept optional explicit `previous_actions`
  - preserve `previous_actions_length` as fallback input
  - enforce that explicit history wins and length mismatch fails
- test modules under `data_creation/tests/` and `tests/games/`
  - add failing tests first for dual-game support and `Escalation_Game` history rules
  - keep regression coverage for resume, artifacts, and constructor validation

## Post-Design Constitution Check

- **Config-truth / no silent fallbacks**: PASS. The only non-ideal prompt reuse is explicit, temporary, and spec-backed. Unsupported games still fail loudly.
- **Real downstream contract validation**: PASS. Both supported games validate through their real scenario constructors.
- **Test-first + regression + mypy**: PASS. The design requires tests before implementation and `mypy` after scenario/pipeline refactors.
- **Reproducible pipelines**: PASS. No artifact bookkeeping or resume features are removed; dual-game support extends the same evidence model.
- **Simplicity**: PASS. One mapping table and one history-normalization rule are enough. No generalized plugin system is justified.

## Complexity Tracking

No constitutional violations or unjustified complexity are required for this feature.
