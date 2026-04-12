<!--
Sync Impact Report
- Version change: 0.0.0 -> 1.0.0
- Modified principles:
  - Placeholder Principle 1 -> I. Config-Truth Research
  - Placeholder Principle 2 -> II. Test-First Change Discipline
  - Placeholder Principle 3 -> III. Loadable Contract Validation
  - Placeholder Principle 4 -> IV. Reproducible Data Pipelines
  - Placeholder Principle 5 -> V. Simplicity And Explicit Failure
- Added sections:
  - Additional Constraints
  - Delivery Workflow
- Removed sections:
  - None
- Templates requiring updates:
  - ✅ updated .specify/templates/plan-template.md
  - ✅ reviewed .specify/templates/spec-template.md (no content change required)
  - ✅ updated .specify/templates/tasks-template.md
  - ✅ reviewed README.md (no constitution-reference change required)
- Follow-up TODOs:
  - None
-->
# LLM Emotional Game Theory Research Constitution

## Core Principles

### I. Config-Truth Research
All experiment, dataset, and transformation code MUST follow the declared config and
declared inputs. Silent fallbacks, implicit provider swaps, hidden default datasets,
or auto-repaired experimental settings are forbidden unless the behavior is explicitly
declared in the feature spec and test coverage. Research artifacts MUST preserve
enough metadata to reconstruct what configuration, prompt assets, model endpoints,
and source data produced the result.

Rationale: In this repository, a convenient fallback is not harmless. It mutates the
experiment and makes later analysis dishonest.

### II. Test-First Change Discipline
Every feature and bug fix MUST start with a failing test or an explicitly documented
test gap that is then closed in the same change. Red-Green-Refactor is the default
workflow. New behavior MUST include the smallest integrated test that proves the
change at the system boundary, and refactors MUST run mypy for modified Python code.
Regression validation is mandatory before completion.

Rationale: This repo already mixes research code, data pipelines, and runtime systems.
Without test-first discipline, breakage hides inside "just scripts."

### III. Loadable Contract Validation
Any feature that produces data for another runtime component MUST validate against the
real downstream contract, not only an approximate local schema. If a dataset is meant
to load through a game class, scenario class, or experiment loader, the implementation
MUST exercise that real loader in tests or validation code. Success artifacts MUST
contain only loadable records unless the spec explicitly states otherwise.

Rationale: The real contract lives where the data is consumed. Duplicate schemas drift.

### IV. Reproducible Data Pipelines
Long-running or high-volume data-processing jobs MUST provide visible progress, resume
support, deterministic identity bookkeeping, and machine-readable success/failure
artifacts. GPU-consuming workflows MUST include monitoring. Every experiment or
transformation run MUST preserve provenance, counters, timestamps, and output paths in
run metadata. Broken rows MUST be accounted for explicitly rather than disappearing.

Rationale: Research pipelines fail in the middle. If they cannot resume or explain
what happened, the output cannot be trusted.

### V. Simplicity And Explicit Failure
KISS and YAGNI are default policy. New abstractions, fallback branches, service layers,
or configuration knobs require a concrete need tied to the feature spec. The system
MUST fail loudly on invalid state, malformed inputs, or unsupported modes rather than
guessing. Shell scripts MUST stay simple. New code files that are scripts MUST begin
with a shebang and a short purpose comment.

Rationale: Most accidental complexity in this repo would come from "flexibility" that
only obscures what the code is doing.

## Additional Constraints

- Data classes MUST not hide required runtime inputs behind field defaults. Defaults
  belong at instantiation sites, not in dataclass definitions, unless there is a
  deliberate and reviewed invariant.
- Backward compatibility is not a default requirement. New work SHOULD implement the
  correct behavior directly unless the feature spec explicitly demands compatibility.
- Documentation is part of delivery. When a change is committed, the nearest README or
  `claude_doc` material MUST be checked and updated if the behavior, workflow, or
  artifact contract changed.
- Guidance documents MUST state intent, avoid overlapping sections, and include only
  information necessary for the document's purpose.

## Delivery Workflow

- Specification work MUST make research-impacting factors explicit: prompt design,
  data format, evaluation method, and other variables that can change the scientific
  meaning of the result.
- Planning work MUST describe how configuration adherence, contract validation,
  resumability, and provenance will be enforced.
- Task generation MUST include tests, validation, documentation updates, and any
  logging/metadata work needed to prove the feature is scientifically sound.
- Implementation work MUST not revert unrelated user changes and MUST prefer the
  smallest coherent design that satisfies the spec.
- Final summaries MUST include concrete evidence that the request was met, such as key
  test coverage, validation logs, or run outputs.

## Governance

This constitution overrides generic defaults in spec-kit templates when they conflict.
Compliance review is required for every plan, task list, implementation, and commit.

Amendment policy:
- MAJOR: Remove or redefine a principle in a way that changes repository governance.
- MINOR: Add a principle or materially expand a mandatory workflow or constraint.
- PATCH: Clarify wording, fix inconsistencies, or improve guidance without changing
  governance intent.

Review policy:
- Constitution compliance MUST be checked during planning and before final delivery.
- Any justified deviation MUST be called out explicitly in the plan under complexity
  or governance rationale.
- Placeholder constitutional text is not acceptable in active repository workflows.

Source of truth policy:
- Runtime guidance in `AGENTS.md` and `CLAUDE.md` may provide more specific operating
  instructions, but they MUST not weaken this constitution.
- When templates and this constitution disagree, the constitution wins and the
  templates MUST be updated.

**Version**: 1.0.0 | **Ratified**: 2026-04-06 | **Last Amended**: 2026-04-06
