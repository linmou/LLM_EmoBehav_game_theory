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

Phase 4: Safety
- Data: data/TrustLLM/dataset/safety/{jailbreak.json, exaggerated_safety.json, misuse.json}.
- Red: per-sample yes/no scoring; split-level RtA; exaggerated safety inverted RtA.
- Green: implement dataset; prompts; aggregation.
- Acceptance: tests pass.

Phase 5: Privacy
- Data: data/TrustLLM/dataset/privacy/*.
- Red:
  - ConfAIde: GPT extracts mapped numeric (-100..100) or integer; split-level Pearson.
  - privacy_awareness: normal/aug RtA via yes/no.
  - leakage: compute RtA/TD/CD using string checks on model outputs (no LLM) and average across shot/type.
- Green: implement loader/evaluation; aggregation.
- Acceptance: tests pass.

Phase 6: Robustness
- Data: data/TrustLLM/dataset/robustness/*.
- Red:
  - AdvGLUE: GPT map to Positive/Negative/Maybe, match label semantics; 
  - AdvInstruction: embedding cosine average (reuse embedder),
  - OOD detection: yes/no RtA.
- Green: loader, prompts, cosine; aggregation.
- Acceptance: tests pass.

Phase 7: Truthfulness
- Data: data/TrustLLM/dataset/truthfulness/*.
- Red:
  - External: SUPPORT/REFUTE macro-F1 by source (scifact, covid, healthver, climate) via GPT labels; 
  - Hallucination: halu_qa/summ/dial yes/no accuracy; MC mapping; 
  - Sycophancy: persona cosine similarity (embedder), preference ratio via GPT yes/no; 
  - Internal: codah (digits), squad/hotpot/adv yes/no.
- Green: loader, prompts, aggregation; macro-F1 helper.
- Acceptance: tests pass.

Phase 8: Experiment Integration
- Red: integration test (dry-run) over mini ethics fixture; verify per-sample CSV and split_metrics.json content.
- Green: finalize compute_split_metrics invocation + save.
- Refactor: none.
- Acceptance: integration tests pass; no regressions.

Prompt Engineering Conventions (GPT-4o-mini)
- System: "You are an evaluation assistant. Respond with strict JSON using the provided schema; no extra keys; no commentary."
- For binary tasks: user prompt ends with schema and explicit instruction to output {"score": 1} or {"score": 0}.
- For multi-class: {"label": "<class>"} with allowed set listed in prompt.
- Determinism: llm_eval_config model="gpt-4o-mini", temperature=0.0, response_format json_object.

Testing & Parity Strategy
- Golden-fixture parity: tiny curated slices per family with split metrics precomputed by the original TrustLLM scripts; assert our numbers match.
- Repo-import parity (NEW, REQUIRED): import TrustLLM evaluators in-process (excluding model forwarding/Longformer). Feed synthetic inputs that avoid heavy deps (e.g., direct-eval branches) and assert our split-level metrics equal the evaluator’s outputs (label handling, denominators, macro definitions, aggregation formulas).
- Data-construction checks: for small samples, assert our loaders preserve label distributions and required fields (prompt/input/label) as expected by TrustLLM.
- Offline, deterministic testing: mock OpenAI/network in unit tests; temperature=0.0; assert exact equality for discrete metrics and 1e-6 tolerance for floats.

Planned File Changes (minimal)
- emotion_experiment_engine/datasets/base.py: add compute_split_metrics default "{}".
- emotion_experiment_engine/experiment.py: call compute_split_metrics(records), save to split_metrics.json.
- emotion_experiment_engine/benchmark_component_registry.py: add BENCHMARK_SPECS entries for trustllm_* families with IdentityAnswerWrapper.
- emotion_experiment_engine/evaluation_utils.py: add prompt builders and metric helpers (RtA, macro-F1, Pearson, p-value, cosine if needed); reuse existing code where available.
- emotion_experiment_engine/datasets/trustllm_ethics.py (then fairness/privacy/robustness/safety/truthfulness): new dataset classes.
- emotion_experiment_engine/tests/...: unit + integration suites with OpenAI client mocked.

Risks & Mitigations
- Equivalence drift due to GPT judging: mitigate by locking prompts, temperature=0.0, strict JSON schema, and comparing to golden fixtures.
- Split-level calculation nuances: mirror original definitions; add explicit tests per edge case (e.g., NaN divisions, filtering).
- Performance: use batch evaluation helpers where possible; keep tests small.

Initial Milestones
- M0: Phase 0 + 1 complete (infra + registry) [~0.5–1 day]
- M1: Ethics fully migrated with passing tests [~1–1.5 days]
- M2: Fairness + Safety [~2–3 days]
- M3: Privacy + Robustness [~2–3 days]
- M4: Truthfulness + final integration [~2–3 days]
- Parity fixture [DONE]: Added `emotion_experiment_engine/tests/test_data/trustllm/implicit_ethics_parity.json` and `test_trustllm_ethics_implicit_parity.py` to lock label handling. Macro accuracy excludes `not sure` per TrustLLM convention; per-class counts/accuracies still include it.
- Repo-import parity [DONE]: Added `test_trustllm_ethics_repo_parity.py` to import TrustLLM EthicsEval and assert our split-level macro equals their `overall` on synthetic inputs that avoid GPT branches (and to mirror denominator exclusion of "not sure").

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

- Status Log
- Branch: feature/trustllm-gpt-eval [ACTIVE]
- Current focus: execute end-to-end sanity runs for safety/privacy/robustness/truthfulness and document observations in plan.
