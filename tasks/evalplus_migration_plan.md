################################################################################
This Penn State College of Information Sciences and Technology system, its equipment, and network connectivity are provided for authorized use according to Penn State Policies AD95, AD96, and ADG02. While the College and University respect users' privacy rights, they also prioritize network safety and security. As a result, IST computer systems may be monitored for lawful purposes as stated in Penn State Policy AD53. This monitoring is conducted to ensure authorized use, manage the system, protect against unauthorized access, and verify the system's security. During monitoring, information, including personal information, may be examined, recorded, copied, and used for authorized purposes. By using this computer system, users consent to monitoring for the listed purposes. Unauthorized use may lead to administrative action or criminal prosecution, as outlined in Penn State Policies.
 
Penn State has a regularly scheduled maintenance window, from 5:00 am – 7:00 am daily, during which time this system may be rebooted or individual services restarted. This may result in temporary interruptions of service.
 
Please note that any data you have stored on this system is not backed up. We recommend that you back up any important data that can’t be recreated somewhere else to ensure that you have a copy.

If you need support or have questions please email helpdesk@ist.psu.edu.
################################################################################
# EvalPlus Migration Plan (Rewritten)

Updated: 2025-10-13 | Commit: TBD

## 1) Goal
Rewrite HumanEval and add MBPP in a unified shape:
- Benchmarks: `humaneval`, `mbpp`
- Tasks: `{default, plus, *}` where `*` means the combination of all supported tasks for that benchmark
- No backward-compat assumptions; keep it simple and explicit (KISS).

## 2) Scope & Assumptions
- Offline-first; no auto-downloads. Require local paths for plus datasets:
  - HumanEvalPlus.jsonl.gz
  - MbppPlus.jsonl.gz
- Use upstream oracles for strict, deterministic scoring:
  - humaneval/default: `human_eval.execution.check_correctness`
  - humaneval/plus: `evalplus.evaluate.check_correctness("humaneval", ...)`
  - mbpp/default: sanitized MBPP checker (EvalPlus helpers if available)
  - mbpp/plus: `evalplus.evaluate.check_correctness("mbpp", ...)`
- Prompting: keep existing HumanEval chat-style wrapper (code-only instruction). Add a minimal MBPP wrapper mirroring it.

## 3) Architecture Changes
- Registry semantics
  - Register explicit tasks: (`humaneval`, `default`), (`humaneval`, `plus`), (`humaneval`, `*`)
  - Same for `mbpp` when implemented
  - Treat `*` as an explicit “combined” task, not a wildcard fallback.
- Factory behavior
  - Remove/warn against automatic fallback from unknown task to `*`. `*` is opt-in (explicit) to mean “combine”.
  - Keep name→dataset-class mapping (one dataset class per benchmark name). Dataset branches by `config.task_type`.

## 4) Dataset Design
### 4.1 HumanEvalDataset (rewrite)
- Loader
  - Modes:
    - default: load original HumanEval (jsonl.gz/jsonl)
    - plus: load HumanEvalPlus (jsonl.gz)
    - *: load HumanEvalPlus once but emit two logical items per task_id (one tagged `mode=default`, one `mode=plus`)
  - Item schema: `BenchmarkItem(id, input_text=row["prompt"], context=None, ground_truth=row, metadata={"entry_point": row["entry_point"], "source": "humaneval|humaneval+", "mode": default|plus})`
- Evaluation
  - default: human_eval checker → 1.0/0.0
  - plus: EvalPlus oracle with cached expected outputs (by dataset hash) → 1.0/0.0
  - *: evaluate per-item according to `metadata.mode`
- Performance
  - Lazy import EvalPlus; cache expected outputs once per run; default conservative timeouts

### 4.2 MBPPDataset (new)
- Loader
  - Modes: default (sanitized MBPP JSON), plus (MbppPlus jsonl.gz), * (emit default+plus logical items per task)
- Evaluation
  - default: sanitized assertions (or EvalPlus helper if available)
  - plus: EvalPlus oracle (`check_correctness("mbpp", ...)`) handling special oracles
- Item schema and `metadata.mode` mirror HumanEvalDataset

## 5) Prompt Wrappers
- HumanEvalPromptWrapper: keep existing chat-style “code only” instruction (parity and clean outputs)
- MBPPPromptWrapper: identical pattern to HumanEval wrapper (system: code-only; user: prompt text)

## 6) Config & Paths
- humaneval/default
  - benchmark.name: humaneval
  - task_type: default
  - data_path: /abs/path/to/HumanEval.jsonl.gz (or .jsonl)
- humaneval/plus
  - task_type: plus
  - data_path: /abs/path/to/HumanEvalPlus.jsonl.gz
- humaneval/*
  - task_type: "*"
  - data_path: /abs/path/to/HumanEvalPlus.jsonl.gz (single source of truth)
- mbpp/default|plus|*
  - analogous; plus and * require MbppPlus.jsonl.gz
- Optional for EvalPlus internals (if we rely on its loaders anywhere):
  - export HUMANEVAL_OVERRIDE_PATH=/abs/path/HumanEvalPlus.jsonl.gz
  - export MBPP_OVERRIDE_PATH=/abs/path/MbppPlus.jsonl.gz

## 7) TDD Plan (Red-Green-Refactor)
- 1) Red
  - `emotion_experiment_engine/tests/unit/datasets/test_humaneval_modes.py`
    - default loads >0; id/prompt/entry_point parity vs original
    - plus loads >0; id/prompt/entry_point parity vs plus jsonl
    - star emits exactly 2× items (default & plus) with `metadata.mode`
  - `emotion_experiment_engine/tests/integration/test_humaneval_evaluation.py`
    - default / plus canonical pass-fail parity; star exposes both views
  - `emotion_experiment_engine/tests/unit/datasets/test_mbpp_modes.py`
    - default/plus/* parity; canonical pass-fail; skips if optional `tree_sitter_python` missing
  - `emotion_experiment_engine/tests/integration/test_mbpp_evaluation.py`
    - canonical pass-fail mirrors unit suite with the same skip guard
  - Ensure tests pick file paths from env vars (HUMANEVAL_PLUS_GZ, HUMANEVAL_ORIG, MBPP_PLUS_GZ). Skip with clear reason if missing.

2) Green
- Rewrite HumanEvalDataset to pass tests (modes, eval, star, caching)
- Add MBPPDataset + wrapper + registry entries; pass tests

3) Refactor
- Remove duplication in mode branching; keep code minimal; annotate tricky parts
- mypy on modified files

4) Regression (回归测试)
- Run full test suite
- Dry-run:
```
python -m emotion_experiment_engine.emotion_experiment_series_runner \
  --config config/humaneval_qwen_series.yaml \
  --name he_star_smoke --dry-run
```

## 8) Registry Changes
- HumanEval (explicit only; no wildcard fallback):
  - ("humaneval", "default"), ("humaneval", "plus"), ("humaneval", "*") → HumanEvalDataset + HumanEvalPromptWrapper
- MBPP (to add):
  - ("mbpp", "default"), ("mbpp", "plus"), ("mbpp", "*") → MBPPDataset + MBPPPromptWrapper
- Update create_benchmark_components: remove automatic fallback to (name, "*")
  - Unknown (name, task_type) should raise KeyError with available combos listed

## 9) Risks & Mitigations
- Untrusted code: rely on EvalPlus sandbox; keep small timeouts; start with small sample_limit
- Path brittleness: tests skip cleanly if env paths not set; dry-run requires explicit paths
- Performance: cache expected outputs; optional `fast_check` in smoke
- Prompt drift: retain code-only wrappers; consider raw pass-through if you want exact parity later

## 10) Checklist
- [x] Rewrite HumanEvalDataset with {default, plus, *} modes
- [x] Add MBPPDataset + MBPPPromptWrapper and register
- [x] Update registry and remove wildcard fallback in `create_benchmark_components`
- [x] Add tests (dataset load, eval, star mode, registry)
- [x] Dry-run configs for humaneval and mbpp
- [ ] Full regression; doc updates

## 11) Open Questions
- Do you want raw pass-through prompts for plus modes (exact txt parity) or keep the current chat-style wrappers? Default is chat-style (cleaner control over outputs).
