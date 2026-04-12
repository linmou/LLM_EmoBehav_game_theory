# Research: Social Game Case Transformation Pipeline

## Decision 1: Implement the feature as a Python CLI under `data_creation/`

- Decision: Add the transformation workflow as a new file-based Python CLI in `data_creation/`, alongside the existing generator scripts.
- Rationale: The repository already treats scenario and dataset generation as script-driven work. Keeping this feature in `data_creation/` reuses existing patterns for progress display, resumability, and test placement without inventing a service layer.
- Alternatives considered:
  - New web service: rejected because the feature is an internal research preprocessing job, not an always-on runtime dependency.
  - Shell-only pipeline: rejected because prompt assembly, JSON validation, resumability, and schema loading are easier to implement and test in Python.

## Decision 2: Use the repository’s existing OpenAI-compatible client pattern for DeepSeek

- Decision: Use the same OpenAI-compatible request style already present in the repository for chat completions, with credentials sourced from `.env`.
- Rationale: The repo already integrates OpenAI-compatible clients and base URLs in multiple places. Reusing that pattern reduces risk and avoids parallel client infrastructure.
- Alternatives considered:
  - Raw HTTP calls: rejected because it duplicates client behavior and increases error-handling surface.
  - LangChain wrapper: rejected because the pipeline only needs direct chat completions and structured parsing, not graph orchestration.

## Decision 3: Use `id + source.game_id` as the resume/dedup identity

- Decision: Treat the top-level `id` plus `source.game_id` as the pipeline identity key for resume, deduplication, and artifact bookkeeping.
- Rationale: The curated `beauty_contest` dataset contains unique top-level `id` values, while `source.game_id` repeats across multiple extracted samples from the same Diplomacy game. Keeping both in the identity rule matches the clarified spec and preserves explicit provenance.
- Alternatives considered:
  - Top-level `id` only: simpler and sufficient for the current file, but rejected because the clarified spec explicitly wants the identity pair.
  - `source.game_id` plus line number: rejected because line numbers are more brittle as a primary identity and do not reflect the curated sample identifier.

## Decision 4: Keep the main output success-only and write failures separately

- Decision: Write only validated transformed cases to the main dataset. Write failed and skipped rows to separate machine-readable artifacts with diagnostics and provenance.
- Rationale: Existing game-loading workflows expect loadable scenario records. Mixing failures into the main dataset would force downstream consumers to learn a new filtering contract and increase experiment contamination risk.
- Alternatives considered:
  - Mixed output with status field: rejected because it weakens the existing loader contract and creates more downstream branching.
  - Abort on first error: rejected because long-running research transformations need best-effort completion and restartability.

## Decision 5: Use direct scenario construction as the contract validation

- Decision: Treat `scenario_class(**data)` as the decisive validation rule for successful transformed rows, using the Beauty Contest scenario class in V1.
- Rationale: The feature’s core requirement is “cases can be loaded by the game class.” The game class itself is the authoritative contract, especially for `previous_actions` validation and any future scenario-class logic that a duplicate local schema would miss.
- Alternatives considered:
  - Custom standalone schema only: rejected because it risks drifting from the real runtime contract.
  - Local schema plus scenario loading as equal authorities: rejected because two contracts invite drift; the game class should win.
  - No scenario loading in the pipeline: rejected because it would defer the most important failure mode to downstream experiments.

## Decision 6: Compose prompts from one shared rubric file plus one pluggable few-shot asset

- Decision: Build the system prompt from `transform_rubrics.md` and add a social-game-specific few-shot asset chosen by explicit mapping for `beauty_contest`.
- Rationale: This directly matches the clarified feature scope while leaving a clean extension point for future games. One shared rubric prevents style drift; one mapped example asset prevents hard-coded prompt text in the script.
- Alternatives considered:
  - Inline prompt strings in the script: rejected because prompt assets are content, not code.
  - One global few-shot file for all games: rejected because game-specific target schemas will differ.

## Decision 7: Preserve run metadata as a first-class artifact

- Decision: Write a run metadata artifact containing input paths, prompt asset references, model identifier, counters, timestamps, and output paths.
- Rationale: The repository is a research repo, and reproducibility matters more than shaving a small amount of file I/O. Explicit metadata is also the cleanest way to support resume and later audit.
- Alternatives considered:
  - Console logs only: rejected because logs are not enough for deterministic rerun or audit.
  - Embed all metadata in every success row: rejected because it bloats the main dataset and repeats run-level data unnecessarily.
