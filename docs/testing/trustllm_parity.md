# TrustLLM Parity Test Matrix

<!-- Updated: 2025-09-25 | commit: pending -->

## Overview

All TrustLLM benchmark families now ship with synthetic fixtures and pytest coverage that mirror the original Longformer/Perspective heuristics while delegating final judgements to GPT based evaluators. Each fixture set sits under `emotion_experiment_engine/tests/test_data/trustllm/parity/<family>/` and feeds unit suites that import TrustLLM stubs to compute the reference metrics.

## Fairness (Phase 8.1)
- Fixtures: `stereotype_recognition_parity.json`, `stereotype_agreement_parity.json`, `stereotype_query_parity.json`, `preference_parity.json`, `disparagement_parity.json`.
- Tests: `test_parity_recognition_against_trustllm`, `test_parity_agreement_against_trustllm`, `test_parity_stereotype_query_against_trustllm`, `test_parity_preference_against_trustllm`, `test_parity_disparagement_against_trustllm`.
- Guarantee: Route short heuristics locally and only invoke GPT fallbacks for ambiguous text, keeping recognition counts, agreement accuracy, RtA refusals, and chi-square outputs aligned with `FairnessEval`.

## Ethics (Phase 8.2)
- Fixtures: `explicit_low_parity.json`, `explicit_high_parity.json`, `implicit_ethics_parity.json`, `social_norm_parity.json`.
- Tests: `test_parity_phase8_explicit_low_against_trustllm`, `test_parity_phase8_explicit_high_against_trustllm`, `test_parity_phase8_implicit_ethics_against_trustllm`, `test_parity_phase8_social_norm_against_trustllm`.
- Guarantee: Macro accuracy (with `not sure` exclusions) and per-class tallies match TrustLLM's ETHICS/Social Norm logic while preserving GPT prompt shapes for open ended cases.

## Privacy (Phase 8.3)
- Fixtures: `privacy_confAide_parity.json`, `privacy_awareness_query_parity.json`, `privacy_leakage_parity.json`.
- Tests: `test_parity_confAide_against_trustllm`, `test_parity_awareness_query_against_trustllm`, `test_parity_leakage_against_trustllm`.
- Guarantee: Pearson correlation, time-to-detection (TD), chance-of-detection (CD), and refusal ratios stay numerically identical to `PrivacyEval`, including GPT replacements for Longformer awareness checks.

## Robustness (Phase 8.4)
- Fixtures: `advinstruction_parity.json`, `advglue_parity.json`, `ood_detection_parity.json`, `ood_generalization_parity.json`.
- Tests: `test_parity_advinstruction_against_trustllm`, `test_parity_advglue_against_trustllm`, `test_parity_ood_detection_against_trustllm`, `test_parity_ood_generalization_against_trustllm`.
- Guarantee: cosine-sim thresholds and RtA refusals replicate `RobustnessEval` outputs, with deterministic GPT fallbacks emulating Longformer labelling.

## Safety (Phase 8.5)
- Fixtures: `jailbreak_parity.json`, `misuse_parity.json`, `exaggerated_safety_parity.json` (≥20 samples per branch).
- Tests: `test_parity_jailbreak_against_trustllm`, `test_parity_misuse_against_trustllm`, `test_parity_exaggerated_against_trustllm`.
- Guarantee: Refusal/overrefusal rates from our GPT judge exactly match TrustLLM `SafetyEval` RtA outcomes, preserving the refusal/compliance split counts for jailbreak and misuse as well as reasonable-rate for exaggerated safety.

## Regression Notes

- Quick parity sweep: `pytest emotion_experiment_engine/tests/test_trustllm_*.py` (family-specific) remains the recommended smoke target.
- Type-checking (`python -m mypy emotion_experiment_engine/datasets/trustllm_safety.py`) currently surfaces pre-existing repository errors outside the TrustLLM parity scope; track in issue #TODO if remediation is required.
