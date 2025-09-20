# TrustLLM Migration Plan (GPT-4o-mini only)

Objective
- Reproduce TrustLLM benchmarks (ethics, fairness, privacy, robustness, safety, truthfulness) inside emotion_memory_experiments using only GPT-4o-mini as evaluator, IdentityAnswerWrapper, and current infra. No Longformer/Perspective, no extra logging/cost tracking. All scoring must be provably equivalent to the original implementation on curated fixtures via TDD.

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

Phase 0: Split-Level Hook (infra)
- Red:
  - tests: dataset implements compute_split_metrics (exists and returns dict of str→float); EmotionExperiment persists split_metrics.json.
- Green:
  - add default method to BaseBenchmarkDataset; integrate call in EmotionExperiment and save JSON.
- Refactor: keep minimal changes; ensure no behavior change for existing memory benchmarks.
- Acceptance: new tests pass; all existing tests green (回归测试).

Phase 1: Registry & Scaffolds
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

Phase 2: Ethics (first vertical)
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

Phase 3: Fairness
- Data: data/TrustLLM/dataset/fairness/*.
- Red:
  - stereotype_agreement (Yes/No JSON), stereotype_recognition (stereotype/anti-stereotype/unrelated JSON), stereotype_query (RtA via yes/no), disparagement (salary p-value), preference (overall/plain/force).
  - split-level tests assert parity with golden fixtures.
- Green:
  - implement loader, GPT prompts, evaluate_response; compute_split_metrics for RtA, counts, p-values (reuse metrics helpers).
- Acceptance: tests pass.

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
- Fixtures: tiny curated slices per family with golden split metrics computed via the original TrustLLM scripts.
- All GPT calls mocked to deterministic JSON; offline CI; assert exact equality for discrete metrics and tolerance (1e-6) for floats.
- Unit tests cover parsing, per-sample evaluation, split aggregation; integration tests exercise EmotionExperiment end-to-end (dry-run).

Planned File Changes (minimal)
- emotion_memory_experiments/datasets/base.py: add compute_split_metrics default "{}".
- emotion_memory_experiments/experiment.py: call compute_split_metrics(records), save to split_metrics.json.
- emotion_memory_experiments/benchmark_component_registry.py: add BENCHMARK_SPECS entries for trustllm_* families with IdentityAnswerWrapper.
- emotion_memory_experiments/evaluation_utils.py: add prompt builders and metric helpers (RtA, macro-F1, Pearson, p-value, cosine if needed); reuse existing code where available.
- emotion_memory_experiments/datasets/trustllm_ethics.py (then fairness/privacy/robustness/safety/truthfulness): new dataset classes.
- emotion_memory_experiments/tests/...: unit + integration suites with OpenAI client mocked.

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

