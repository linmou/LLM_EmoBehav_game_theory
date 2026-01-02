<!--
Updated: 2026-01-01
Purpose: Track incremental fixes for PD defection vectors + delta activations.
Preferences: KISS/YAGNI, TDD (Red-Green-Refactor), no backward-compat fallbacks.
-->

# Fixation Plan: PD Defection Vectors + Delta Activations

This file lists discrete fixes requested for `auto_experiments/task_similarity/` and how we’ll implement/verify them one-by-one.

## Requested changes (in order)

1. **Fix `_decision_rate` padding bug**
   - Problem: uses `logits[:, -1, :]` which can point at PAD position when prompts are padded.
   - Fix: gather logits at last non-pad token index via `attention_mask.sum(1) - 1`.
   - Verification: unit test with padded batch where correct decision depends on last non-pad token.

2. **Delta activation: measure all layers**
   - Change `compute_pd_delta.py` to compute baseline/steered/delta final-token hidden state for **every transformer layer** (0..num_layers-1), not only the last layer.
   - Keep **final-token** hidden state (not mean pooling) as requested.
   - Verification: unit test asserts saved `.npz` contains keys for all layers and deltas computed.

3. **Dataset handling stays “raw text”**
   - Keep probes as opaque strings; no formatting/substitution in `compute_pd_delta`.
   - (Later) dataset module can provide pre-rendered text.

4. **Unify path naming**
   - Replace `task-similarity` (hyphen) with `task_similarity` (underscore) in:
     - docstrings (Responsible paths)
     - CLI defaults (`--output_dir`, etc.)

5. **Unify output layout for PD training**
   - Move `result.json` and `best_vector.npy` into:
     - `output_dir/<model>/<timestamp>/seed_<seed>/`
   - Keep vectors and split manifest in the same run directory.
   - Update smoke test expectations accordingly.

## Notes / open questions

- `compute_pd_delta.py` docs in `auto_experiments/task_similarity/doc/component_compute_pd_delta.md` currently describe mean-pooled per-layer collection; after code changes we should update docs to match “final-token + all layers” behavior (unless you want docs left as historical record).

