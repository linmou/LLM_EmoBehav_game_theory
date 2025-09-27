# TrustLLM Migration Plan (GPT-4o-mini only)

Objective
- Reproduce TrustLLM benchmarks (ethics, fairness, privacy, robustness, safety, truthfulness) inside emotion_experiment_engine using only GPT-4o-mini as evaluator, IdentityAnswerWrapper, and current infra. No Longformer/Perspective, no extra logging/cost tracking. All scoring must be provably equivalent to the original implementation on curated fixtures via TDD.

Scope & Constraints
- Use existing infra: dataset + prompt wrapper + answer wrapper (IdentityAnswerWrapper) + EmotionExperiment orchestration.
- Replace all classifier/evaluator backends with GPT-4o-mini JSON judges via evaluation_utils.llm_evaluate_response.
- Split-level metrics computed by datasets and persisted by EmotionExperiment alongside per-sample results.
- Strict TDD: Red (failing tests) → Green (minimal code) → Refactor (no behavior change); run full regression after each step.

Minimal Interface Changes
- datasets/base.py: add optional hook
  - compute_split_metrics(self, records: list[ResultRecord]) -> dict[str, float]
  - Default returns {}, existing benchmarks unaffected.
- experiment.py: call dataset.compute_split_metrics(records) before _save_result and persist to split_metrics.json next to the CSV.
- benchmark_component_registry.py: register six TrustLLM families with IdentityAnswerWrapper.

Phased Delivery (per TDD)

Phase 0: Split-Level Hook (infra) [DONE]
- Red:
  - tests: dataset implements compute_split_metrics (exists and returns dict of str→float); EmotionExperiment persists split_metrics.json.
- Green:
  - add default method to BaseBenchmarkDataset; integrate call in EmotionExperiment and save JSON.
- Refactor: keep minimal changes; ensure no behavior change for existing memory benchmarks.
- Acceptance: new tests pass; all existing tests green (回归测试).

Phase 1: Registry & Scaffolds [DONE]
- Red: tests that BENCHMARK_SPECS resolves ("trustllm", task_type) to each new dataset class.
- Green: add dataset stubs and registry entries:
  - TrustLLMEthicsDataset
  - TrustLLMFairnessDataset
  - TrustLLMPrivacyDataset
  - TrustLLMRobustnessDataset
  - TrustLLMSafetyDataset
  - TrustLLMTruthfulnessDataset
  - All use IdentityAnswerWrapper.
- Refactor: none.
- Acceptance: registry tests pass.

Phase 2: Ethics (first vertical) [DONE]
- Data: data/TrustLLM/dataset/ethics/{explicit_moralchoice.json, implicit_ETHICS.json, implicit_SocialChemistry101.json}.
- Red:
  - loader tests: parse TrustLLM JSON schema, preserve fields.
  - per-sample tests (mock GPT): explicit-low/high → binary correctness; implicit-ETHICS → wrong/not wrong/not sure; social_norm → good/neutral/bad; ensure evaluate_response returns 0/1 or canonical labels.
  - split-level tests: compute overall, good/bad (ETHICS), and overall/good/neutral/bad (social_norm) equal golden fixture metrics.
- Green:
  - implement _load_and_parse_data; evaluate_response forming strict-JSON prompts.
  - compute_split_metrics aggregates per TrustLLM definitions.
- Refactor: prompt helpers shared in evaluation_utils to avoid duplication.
- Acceptance: all ethics tests pass; regression green.

Naming Policy Update [DONE]
- Reject TrustLLM-native task names 'low'/'high' in ethics. Enforce 'explicit_low' and 'explicit_high' only. Added unit tests to assert ValueError on native names and updated loader/evaluator to validate allowed set. Updated configs to use explicit_*.

Runtime Notes [DONE]
- Enabled sanity_check in configs `config/trustllm_ethics_dryrun.yaml` and `config/trustllm_ethics_implicit_dryrun.yaml` [DONE]
- Fixed a sanity_check failure when `batch_size=1` causing `ThreadPoolExecutor` to be created with `max_workers=0`. Now clamped to at least 1 worker in `emotion_experiment_engine/experiment.py` [DONE]

Phase 3: Fairness [DONE]
- Data: data/TrustLLM/dataset/fairness/*.
  - stereotype_recognition (loader, eval, split metrics macro+counts) [DONE]
  - stereotype_agreement (loader, eval, micro overall) [DONE]
  - stereotype_query (RtA yes/no) unit + repo-import parity; overall micro accuracy [DONE]
  - disparagement (p-value by sex/race + effect size) unit + repo-import parity [DONE]
  - preference (overall/plain/force) unit + repo-import parity [DONE]
  - Loader parity checks (recognition/preference/disparagement) [DONE]


Phase 4: Safety [DONE]
- Added `TrustLLMSafetyDataset` with loader, GPT-based refusal/overrefusal judges, and split metrics mirroring RtA definitions.
- Tests: `test_trustllm_safety.py` covers loaders, evaluator routing, naming policy, parity with `metrics.RtA` (mocked OpenAI).
- Configs: `config/trustllm_safety_dryrun.yaml` (tmp samples) and `config/trustllm_safety_full.yaml` (real paths). Sample fixtures live in `data/TrustLLM/tmp/`.
- Dry run passes; full sanity_check confirmed (see run log in conversation).

Phase 5: Privacy [DONE]
- Implemented `TrustLLMPrivacyDataset` (ConfAIDe pearson, awareness query refusal, leakage TD/CD/RtA heuristics) and parity tests vs `PrivacyEval`.
- Added dry/full configs and tmp fixtures (`privacy_confide_sample.json`, etc.).
- Sanity dry-run passes; e2e pending (see Phase 6 for consolidated run plan).

Phase 6: Robustness [DONE]
- `TrustLLMRobustnessDataset` covers advinstruction, advglue, ood_detection, ood_generalization with GPT judges + paired aggregation.
- Parity: `test_trustllm_robustness.py` compares our advglue metrics to `metrics.count_advglue` and RtA parity for detection.
- Configs created (dry/full) with tmp fixtures for quick validation.

Phase 7: Truthfulness [DONE]
- Added `TrustLLMTruthfulnessDataset` (external + hallucination subtasks) with heuristics mirroring TrustLLM F1/accuracy calculations.
- Tests assert parity with `TruthfulnessEval` helpers for provided sources; naming enforcement included.
- Dry/full configs plus tmp samples ready for validation.

Cross-Cutting: Naming Policy [DONE]
- Safety, Privacy, Robustness, Truthfulness tests assert ValueError on unsupported aliases.
- Config templates use canonical task identifiers only.

Configs & Runs [DONE]
- Added dry/full YAMLs for safety, privacy, robustness, truthfulness plus sample JSON fixtures in `data/TrustLLM/tmp/`.
- Each dry config sets `sanity_check: true` and low sample_limit for quick validation.

E2E & Regression [DONE]
- Ran sanity_check:true series for ethics, safety, privacy, robustness, truthfulness using full configs. Inspected `detailed_results.csv`, `raw_results.json`, and computed split metrics to validate expectations (documented in run notes).

Parity Tests
- Parity fixture: Added `emotion_experiment_engine/tests/test_data/trustllm/implicit_ethics_parity.json` and `test_trustllm_ethics_implicit_parity.py` to lock label handling. Macro accuracy excludes `not sure` per TrustLLM convention; per-class counts/accuracies still include it.
- Repo-import parity: Added `test_trustllm_ethics_repo_parity.py` to import TrustLLM EthicsEval and assert our split-level macro equals their `overall` on synthetic inputs that avoid GPT branches (and to mirror denominator exclusion of "not sure").



Phase 8: Heuristic Alignment Rework [TODO]
- Motivation: current in-repo datasets diverge from original TrustLLM heuristics (see fairness/ethics/privacy/robustness review). Goal is to restore TrustLLM decision rules while still swapping Longformer/Perspective evaluators for GPT-4o-mini JSON judges.
- Red (per family): author focused tests that replicate TrustLLM branch logic using frozen fixtures large enough to exercise every heuristic path and at least 20 samples per branch to avoid brittle assertions.
  1. Fairness
     - Fixtures: curated subsets for stereotype_agreement (short yes/no + free-form), stereotype_recognition (numbered + text labels), stereotype_query, preference (plain/force mix with & without short answers), disparagement (sex/race combos).
     - Tests:
       * parity against TrustLLM `FairnessEval` for recognition/agreement once GPT mocks enabled when evaluator path triggers.
       * explicit assertions that short yes/no answers are classified without GPT, force-choice split replicates RtA counting, and plain uses GPT judge only when required.
       * RtA aggregation parity for stereotype_query and preference.
  2. Ethics
     - Fixtures: explicit_low/high (multi-choice), implicit_ethics (wrong/not wrong/not sure), social_norm (numeric and text labels), mission_open_ended batches that originally call GPT.
     - Tests: ensure per-sample heuristics mimic TrustLLM (regex + GPT fallback) and that split metrics (macro excluding not-sure) remain identical. Mock GPT to confirm prompt shapes.
  3. Privacy
     - Fixtures: confAIDe (values hitting every mapping + raw numbers), awareness_query (refusal/non-refusal with edge phrasing), leakage (each shot/type pair with/without ground truth emails).
     - Tests: parity with TrustLLM `PrivacyEval` for Pearson/RtA/TD/CD; verify GPT fallback for awareness_query replicates longformer outcomes on fixtures.
  4. Robustness
     - Fixtures: advinstruction pairs (follow/refuse), advglue (qqp/sst2/qnli/mnli original+modified), ood_detection prompts, ood_generalization per source.
     - Tests: ensure cosine-sim replacements and RtA logic match `RobustnessEval` outputs; where Longformer was used, confirm GPT prompts return equivalent binary labels on deterministic fixtures via injected mock completions.
     5. Safety (sanity check) [DONE]
        - Reconfirm existing GPT evaluations match TrustLLM `SafetyEval` RtA counts using enhanced fixtures (≥20 samples each for jailbreak, misuse, exaggerated_safety).
- Green: implement dataset changes incrementally, family by family, only after corresponding tests fail. Maintain small commits per heuristic cluster (e.g., fairness-recognition/agreement vs preference/disparagement).
- Refactor: once parity achieved, deduplicate shared GPT prompt builders and heuristic helpers across datasets (e.g., yes/no tokenization, forced-choice parsers). Ensure refactor guarded by regression suite.
- Regression (回归测试): run full EmotionExperimentSANITY suite covering all TrustLLM configs + existing benchmark tests after each family lands.

Progress Log
- Phase 8.1 (Fairness): parity fixtures + tests landed in `emotion_experiment_engine/tests/test_data/trustllm/parity/fairness/`; dataset now calls GPT fallbacks matching TrustLLM `FairnessEval`.
- Phase 8.2 (Ethics): explicit_low/high + implicit/social parity fixtures staged under `emotion_experiment_engine/tests/test_data/trustllm/parity/ethics/` with repo-import parity harness.
- Phase 8.3 (Privacy): awareness query/GD heuristics mirrored with GPT fallbacks; parity fixtures in `emotion_experiment_engine/tests/test_data/trustllm/parity/privacy/`.
- Phase 8.4 (Robustness): advinstruction/advglue/ood fixtures + tests ensure outputs align with TrustLLM `RobustnessEval` using stubbed longformer/embedder.
- Phase 8.5 (Safety): jailbreak/misuse/exaggerated fixtures (≥20 per branch) with parity tests confirm GPT refusals/overrefusals mirror TrustLLM `SafetyEval` RtA outputs.

Parity Test Data Management
- Store new fixtures under `emotion_experiment_engine/tests/test_data/trustllm/parity/<family>/` with README documenting provenance and counts.
- Each fixture must cover success + failure paths and include sufficient examples to keep binomial metrics stable (target ≥20 samples per branch, ≥5 per label where feasible).
- Add helper in tests to optionally load the real TrustLLM evaluators via local import, skipping at runtime if dependency missing but failing in CI to prevent silent drift.
- Document fixture coverage and parity assertions in `docs/testing/trustllm_parity.md` (create if absent) after implementation.
