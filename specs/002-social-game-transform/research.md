# Research: Social Game Case Transformation Pipeline

## Decision 1: Keep the feature as a Python CLI under `data_creation/`

- Decision: Continue implementing the transformation workflow as a file-based Python CLI in `data_creation/`.
- Rationale: The repository already treats scenario and dataset generation as script-driven research work. This keeps the feature close to the existing tests, prompt assets, and artifact-writing patterns without inventing a service layer.
- Alternatives considered:
  - Web service: rejected because this is offline preprocessing, not an always-on runtime dependency.
  - Shell-only pipeline: rejected because prompt assembly, JSON normalization, and contract validation are cleaner and more testable in Python.

## Decision 2: Use explicit per-game mapping, not auto-discovery

- Decision: Represent supported social games with an explicit mapping that carries the target scenario class, canonical target game name, artifact filenames, prompt wording, and deterministic field injectors.
- Rationale: The current code is already mapping-oriented. A small explicit mapping is still the minimum change that keeps unsupported modes loud and predictable.
- Alternatives considered:
  - Auto-discover from `games/`: rejected because prompt behavior and artifact contracts are part of the feature, not just Python imports.
  - Registry framework: rejected because it adds abstraction without a concrete third-phase need.

## Decision 3: Keep real scenario-constructor validation as the success boundary

- Decision: Preserve `scenario_class(**data)` as the decisive validation rule for success rows.
- Rationale: The feature exists to produce rows that downstream game code can load directly. The scenario class is the real contract and catches shape and semantic mismatches that a duplicate local schema would miss.
- Alternatives considered:
  - Standalone local schema only: rejected because it would drift from the real consumer.
  - Loose shape checks plus later runtime validation: rejected because it delays the real failure until downstream experiments.

## Decision 4: Preserve success-only output with explicit failure and skip artifacts

- Decision: Keep the main success dataset limited to loadable rows and continue writing failed and skipped rows to separate machine-readable artifacts.
- Rationale: Downstream loaders should never need to filter mixed-status rows. The repo constitution also requires explicit accounting of broken rows.
- Alternatives considered:
  - Mixed output with status flags: rejected because it weakens the loadable-contract boundary.
  - Abort on first bad row: rejected because long-running jobs must continue over valid rows.

## Decision 5: Few-shot eligibility must stay inside the same social game and current run variants

- Decision: For a run, only consider few-shot examples from the selected social game and only from variants that actually appear in the current run input.
- Rationale: The raw structured source data already carries the structural diversity the user wants. The transform stage should not introduce irrelevant variant language or cross-game wording that creates lexical shortcuts.
- Alternatives considered:
  - Allow all variants from the same game: rejected because it injects off-run wording that does not help the current batch.
  - Allow cross-game examples for more lexical variety: rejected because it violates config-truth and muddies the prompt contract.

## Decision 6: Build few-shot packs per source row, not one shared pack per run

- Decision: Construct a separate few-shot pack for each source row.
- Rationale: The user wants same-variant emphasis with controlled cross-variant diversity. That rule only makes sense at the row level, because the target row's own variant is the anchor.
- Alternatives considered:
  - One shared pack for the whole run: rejected because it cannot express "mostly same variant" for every row when the run contains multiple variants.
  - Randomized pack selection per worker: rejected because it weakens reproducibility.

## Decision 7: Each per-row pack must contain exactly 2 cross-variant examples

- Decision: For each source row, reserve exactly 2 few-shot slots for examples from other run-present variants, and fill every remaining slot from the source row's own variant.
- Rationale: This converts the user's "mainly same-variant, plus 2 others" guidance into a hard rule that can be tested and audited.
- Alternatives considered:
  - Require coverage of every run-present variant: rejected because it can over-constrain the pack and dilute same-variant emphasis.
  - Let the scorer decide any mix: rejected because it drifts from the requested composition.

## Decision 8: Lexical diversity scoring should only see `description` and `behavior_choices`

- Decision: Compute few-shot diversity scores from each example's `description` and `behavior_choices` text only.
- Rationale: The user explicitly wants transform-stage diversity pressure on n-grams, not structural JSON fields or provenance boilerplate.
- Alternatives considered:
  - `description` only: rejected because behavior-choice phrasing is a major source of lexical shortcuts.
  - Full serialized JSON: rejected because repeated scaffolding from keys and metadata would dominate the score.

## Decision 9: Use a deterministic greedy weighted 3/4/5-gram gain rule

- Decision: Rank eligible examples by a greedy score that rewards newly added 3-grams, 4-grams, and 5-grams and penalizes repeated 3-grams, 4-grams, and 5-grams against already selected examples.
- Rationale: This directly optimizes the actual objective: increasing classic n-gram diversity while suppressing repeated scaffolding. It is also deterministic and cheap to audit.
- Alternatives considered:
  - TF-IDF similarity only: rejected because it is too indirect for repeated scaffold control.
  - Random sampling: rejected because it is weaker, noisier, and not reproducible enough for this research repo.

## Decision 10: Fail loudly when the few-shot pool cannot satisfy the pack rule

- Decision: Treat insufficient same-variant or cross-variant pools as explicit setup or row-selection failures instead of silently relaxing the composition rule.
- Rationale: Silent relaxation would mutate the prompt design and make later comparisons dishonest.
- Alternatives considered:
  - Backfill missing same-variant slots with more cross-variant examples: rejected because it changes the requested composition.
  - Drop the row and continue without clear accounting: rejected because it hides a feature-level configuration problem.

## Decision 11: Keep candidate and diversity artifacts as classic audit outputs

- Decision: Preserve candidate-generation support and diversity reports as supporting artifacts, and extend them so few-shot selection remains auditable rather than opaque.
- Rationale: The current work already measures classic n-gram diversity. The new selection policy should add auditability, not reduce it.
- Alternatives considered:
  - Keep only final selected outputs: rejected because it makes it harder to understand why diversity changed.
  - Switch to embedding-heavy evaluation: rejected because the user explicitly asked to focus on n-gram diversity in this stage.
