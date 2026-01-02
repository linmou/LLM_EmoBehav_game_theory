# Implementation Plan: Prisoner's Dilemma Emotion Steering Similarity

**Branch**: `002-pd-steering-sim` | **Date**: 2025-12-08 | **Spec**: specs/002-pd-steering-sim/spec.md  
**Input**: Feature specification from `specs/002-pd-steering-sim/spec.md`

## Summary

Implement a config-driven analysis pipeline that reuses the existing `emotion_experiment_engine` game-theory benchmark and emotion steering vectors to measure how layer-level hidden states move toward precomputed Prisoner's Dilemma (PD) defection directions.  
The pipeline will:
- identify switcher vs non-switcher samples from the existing game-theory PD raw results JSON,  
- run or reuse PD benchmark runs with layer-level emotion steering for Qwen2.5-1.5B-Instruct,  
- capture hidden states at the last input token for each layer with and without steering,  
- compute cosine similarity to PD defection vectors per layer,  
- compare similarity shifts for switchers vs non-switchers, and  
- aggregate similarity changes by emotion and intensity.

## Technical Context

**Language/Version**: Python 3.10+ in `llm_fresh` conda environment  
**Primary Dependencies**: PyTorch, vLLM, Transformers, numpy, PyYAML, existing `emotion_experiment_engine` and `auto_experiments` modules  
**Storage**: Local filesystem (YAML configs, JSON raw results, NumPy/pt steering vectors, CSV/JSON summary outputs)  
**Testing**: pytest for unit/integration tests, mypy for type-checking new modules  
**Target Platform**: Linux server with CUDA GPU (same environment used for existing Qwen/vLLM experiments)  
**Project Type**: Single-project research/CLI pipeline within existing repository  
**Performance Goals**: Complete one PD similarity analysis run (single model, several emotions) within a few GPU hours; no interactive latency requirements  
**Constraints**: Reuse existing steering-vector loading and benchmark runners; avoid new services or frameworks; keep memory footprint bounded by processing per-batch and per-layer rather than loading all samples into RAM at once  
**Scale/Scope**: Focus on Qwen2.5-1.5B-Instruct and the existing PD game-theory dataset (hundreds–low thousands of samples), with support for multiple emotions and intensities via YAML config

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Test-First (TDD): New functionality will be driven by pytest tests (unit for similarity/math utilities, integration for PD similarity runs) before implementing the full pipeline.  
- Simplicity / YAGNI: Implementation will add a small number of focused modules under `auto_experiments/layer_vector_sim` and reuse existing loaders/hooks instead of introducing new frameworks or services.  
- Observability: Analysis will log config parameters, sample counts, and key summary statistics so runs are debuggable from logs alone.  

Gate Status: **PASS** – No constitution violations identified or requested.

## Project Structure

### Documentation (this feature)

```text
specs/002-pd-steering-sim/
├── spec.md         # Feature specification
├── plan.md         # This file (/speckit.plan command output)
├── research.md     # Phase 0 output (design decisions & rationale)
├── data-model.md   # Phase 1 output (entities and relationships)
├── quickstart.md   # Phase 1 output (how to run the analysis)
├── contracts/      # Phase 1 output (interface/contract description)
└── tasks.md        # Phase 2 output (/speckit.tasks command - not created here)
```

### Source Code (repository root)

```text
auto_experiments/
  layer_vector_sim/
    specs/002-pd-steering-sim/        # Feature docs
    # [to be added] pd_steering_similarity analysis module(s)

emotion_experiment_engine/
  benchmark_component_registry.py     # Game-theory benchmark registry
  # Existing EmotionExperiment and steering-vector loading logic

results/
  new_game_theory/
    Qwen2.5-1.5B-Instruct_game_theory_Prisoners_Dilemma_.../raw_results.json

auto_experiments/
  task_similarity/
    results/steering_vectors/.../layer_vectors/  # Existing PD defection layer vectors

tests/
  auto_experiments/
    # [to be added] test_pd_steering_similarity.py
```

**Structure Decision**: Reuse the existing monorepo layout; implement PD steering similarity analysis under `auto_experiments/layer_vector_sim`, read benchmark data and steering vectors from existing result directories, and add tests under `tests/auto_experiments`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| *(none)*  | Not applicable for this feature | Existing repo structure already supports required scope |

