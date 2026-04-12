# Implementation Plan: Social Game Case Transformation Pipeline

**Branch**: `002-social-game-transform` | **Date**: 2026-04-06 | **Spec**: [spec.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/spec.md)
**Input**: Feature specification from `/specs/002-social-game-transform/spec.md`

**Note**: This plan is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a resumable Python CLI under `data_creation/` that transforms curated `beauty_contest` social-game JSONL rows into a success-only scenario dataset for the Beauty Contest game. The pipeline uses DeepSeek chat completions with a shared rubric plus pluggable few-shot examples, validates successful outputs by direct scenario construction with `scenario_class(**data)`, preserves provenance and the `id + source.game_id` identity pair, and writes separate failure/skip artifacts plus run metadata for reproducibility.

## Technical Context

**Language/Version**: Python 3.10+ in the repository conda environment used for data-generation scripts (`llm` on the current machine)  
**Primary Dependencies**: Standard library (`argparse`, `json`, `pathlib`, `concurrent.futures`, `datetime`), `python-dotenv`, OpenAI-compatible `openai` client stack already used in-repo, existing `games/beauty_contest.py` scenario class  
**Storage**: Local files: source JSONL input, success dataset JSON, failure/skipped artifact JSONL, run metadata JSON  
**Testing**: `pytest`, targeted integration tests for CLI output and scenario loading, mypy on modified Python modules  
**Target Platform**: Linux development and research environment in this repository  
**Project Type**: Internal research data-processing CLI pipeline  
**Performance Goals**: Process the full `beauty_contest` curated file with visible progress, resumability, and deterministic row accounting while keeping external API usage explicit and bounded  
**Constraints**: No silent fallback logic, success-only main dataset, identity is `id + source.game_id`, first release limited to `beauty_contest`, prompt contract must incorporate `transform_rubrics.md` plus pluggable few-shot assets, preserve provenance and restartability, contract validity is decided by direct scenario construction  
**Scale/Scope**: Initial scope is the curated `beauty_contest` dataset at `/home/jjl7137/diplomacy_cicero/social_game_outputs/beauty_contest/curated_cases/beauty_contest_cases.jsonl`; design must leave extension points for additional social games with explicit mapping and assets

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Config-truth preserved: the design uses explicit social-game mapping, explicit prompt assets, explicit `.env` credentials, and no silent provider or dataset fallback.
- Real downstream contract preserved: successful transformed rows are validated by direct construction of the target scenario class, `scenario_class(**data)`, rather than by a duplicate local schema alone.
- Test-first discipline preserved: the implementation requires failing tests first, targeted regression checks, and mypy on modified Python code.
- Reproducible pipeline behavior preserved: visible progress, resume behavior, provenance retention, failure/skipped artifacts, and run metadata are all part of the design.
- Simplicity preserved: one file-based CLI pipeline under `data_creation/`, no service layer, no generalized registry beyond an explicit supported-game mapping.

**Gate Result (pre-research)**: PASS.  
**Gate Result (post-design)**: PASS.

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
├── README.md
├── transform_social_game_cases.py
└── tests/
    ├── features/
    │   └── social_game_transform.feature
    └── test_transform_social_game_cases.py

games/
├── beauty_contest.py
└── game_configs.py

tests/
└── games/
    └── test_beauty_contest_game_config.py

beauty_contest_few_shot_examples.json
transform_rubrics.md
```

**Structure Decision**: Extend the existing `data_creation/` script area with one transformation CLI and focused tests. Reuse the real Beauty Contest scenario contract in `games/beauty_contest.py` instead of maintaining a second authoritative schema layer.

## Phase 0: Research Outcome

Phase 0 outputs are documented in [research.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/research.md). The resolved planning decisions are:
- Client pattern: use the repository’s existing OpenAI-compatible client style for DeepSeek chat completions.
- Artifact strategy: one success dataset plus separate failure/skipped artifacts and run metadata.
- Resume strategy: deduplicate by `id + source.game_id`.
- Contract strategy: treat direct scenario construction as the decisive validation boundary.
- Placement: implement in `data_creation/` with pytest coverage.

## Phase 1: Design & Contracts

Phase 1 outputs:
- [data-model.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/data-model.md)
- [quickstart.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/quickstart.md)
- [social-game-transform-cli.md](/home/jjl7137/LLM_EmoBehav_game_theory_real_flexible_dataset/specs/002-social-game-transform/contracts/social-game-transform-cli.md)

Design summary:
- A single CLI command transforms one supported social game at a time.
- V1 supports only `beauty_contest`, with extension reserved for future explicit mappings and prompt assets.
- Prompt construction composes the shared rubric file with one social-game-specific few-shot asset.
- A transformed row is considered contract-valid only if the target scenario class accepts it through direct construction.
- Output splits into success dataset, failure artifact, skipped artifact, and run metadata so downstream experiments only consume loadable cases.

## Post-Design Constitution Check

- Simplicity preserved: one CLI pipeline, file-based artifacts, no new runtime boundary.
- Research robustness preserved: explicit identity, provenance retention, no fallback identity heuristics, no silent downgrade from invalid records.
- Contract correctness preserved: successful output is defined by the real downstream game class.
- Validation discipline preserved: tests, regression checks, and mypy remain part of the delivery path.

**Result**: PASS

## Complexity Tracking

No constitution exceptions or complexity waivers are required. The design stays inside one CLI module, one focused prompt-pack mapping, and explicit artifact files.
