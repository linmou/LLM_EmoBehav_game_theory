# PD Self-Report General Sweep (Qwen2.5-0.5B)

## Intent
Build a full self-report logprob sweep across all hidden layers before choosing Prisoner's Dilemma follow-up slices, so layer and intensity selection is driven by measured self-report response rather than inherited PD-aligned guesses.

## Research Questions
1. For each emotion, which hidden layers increase self-report target probability most reliably?
2. Does the best self-report layer differ from the previously used PD-aligned layer?
3. Can we use self-report sweep results to shortlist layer and alpha settings for a later PD generation sweep with lower intensities?
4. Given the current evidence, is coupled movement between self-report target probability and PD defection likely to be emotion-specific rather than universal?

## Fixed Design For This Iteration
- model: `/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct`
- self-report task: `emotion_check/self_report_emotion_options6`
- self-report option set: `anger, happiness, sadness, fear, disgust, surprise, neutral`
- steer emotions: `anger, happiness, sadness, fear, disgust, surprise`
- self-report layers: full hidden-layer sweep
- self-report intensities: `1, 2, 4, 6, 8, 10, 15, 20, 40, 80`
- steering position: last token only
- option shuffle: `per_item`, seed `42`
- decoding / score mode: prompt logprob, deterministic
- parse-failure gate for later PD study: `unknown_ratio <= 0.10`

## Changed Factors That May Impact Interpretation
- New experiment folder separates the general self-report sweep from the older PD-coupling folder.
- Self-report uses a broad intensity grid, while PD will remain on a lower intensity grid in the next iteration.
- Layer selection is no longer inherited from the older PD-aligned slice list.

## Hypothesis
- The old PD-aligned layers are suboptimal for at least `fear` and `disgust`, because those slices barely moved `p_target`.
- Strong self-report layers will cluster in a limited band rather than spread uniformly across all 24 layers.
- The best self-report alpha values will usually be higher than the acceptable PD-generation alpha values.

## Plan
1. Add a new general-sweep runner that reuses one loaded model instead of launching one process per `(emotion, layer, alpha)`.
2. Run the full self-report sweep across all layers and the requested intensity grid.
3. Aggregate a ranking table per emotion, layer, and intensity.
4. Compare the top self-report layers against the earlier PD-aligned layers.
5. Record shortlist candidates for the next PD generation iteration.

## Reproduction Skeleton
```bash
conda run -n llm python -m pytest tests/test_pd_selfreport_general_sweep.py
conda run -n llm python auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_general_selfreport_sweep.py
conda run -n llm python auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analyze_general_selfreport_sweep.py
```

## Iteration 1

### Status
- completed

### Implementation Notes
- Added a dedicated general-sweep runner that:
  - loads the model once
  - computes a shared neutral baseline once
  - sweeps all `24` hidden layers for all `6` emotions and all `10` requested intensities
  - writes per-condition artifacts with metadata for resume-safe execution
- Fixed a neutral-reference bug:
  - baseline `p_target` must be computed against the same target emotion under neutral steering
  - otherwise `delta_p_target_mean` is wrong or empty
- Refactored the runner batching:
  - first version called vLLM once per prompt
  - current version batches all prompt-target pairs per condition
  - this materially reduced the remaining wall-clock time during the resumed full run
- Added a comparison script to merge:
  - full self-report sweep results
  - previous PD-aligned self-report slice results
  - previous PD generation summaries

### Commands
```bash
conda run -n llm python -m pytest \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_selfreport_general_sweep_compare.py \
  tests/test_pd_selfreport_coupling_runner.py \
  tests/test_pd_selfreport_coupling_analysis.py \
  tests/test_emotion_experiment_engine_import_safe.py

conda run -n llm python \
  auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_general_selfreport_sweep.py

conda run -n llm python \
  auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analyze_general_selfreport_sweep.py

conda run -n llm python \
  auto_experiments/pd_selfreport_general_sweep_qwen25_05b/compare_with_previous_pd.py
```

### Key Artifacts
- full self-report outputs:
  - `results/auto_experiments/pd_selfreport_general_sweep_qwen25_05b/self_report_logprob/`
- ranked self-report summaries:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/best_by_emotion.csv`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/top5_by_emotion.csv`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/previous_pd_layer_comparison.csv`
- crosswalk against prior PD:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/selfreport_vs_previous_pd_crosswalk.csv`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/next_pd_shortlist.csv`

### Main Results
- full sweep completion:
  - `1440 / 1440` conditions
  - `6 emotions × 24 layers × 10 intensities`
- best self-report slices by emotion:
  - `anger`: `layer 17`, `alpha 80`, `delta_p_target_mean = 0.3469`
  - `disgust`: `layer 14`, `alpha 6`, `delta_p_target_mean = 0.0434`
  - `fear`: `layer 15`, `alpha 15`, `delta_p_target_mean = 0.9158`
  - `happiness`: `layer 20`, `alpha 40`, `delta_p_target_mean = 0.8117`
  - `sadness`: `layer 16`, `alpha 40`, `delta_p_target_mean = 0.9214`
  - `surprise`: `layer 24`, `alpha 80`, `delta_p_target_mean = 0.9063`
- comparison against the old PD-aligned layers:
  - `anger`: old layer `17` was already the best self-report layer
  - `disgust`: old layer `21` was poor; best layer moved to `14`
  - `fear`: old layer `23` was poor; best layer moved to `15`
  - `happiness`: old layer `17` was suboptimal; best layer moved to `20`
  - `sadness`: old layer `15` was close but not best; best layer moved to `16`
  - `surprise`: old layer `15` was poor; best layer moved to `24`
- gains over previous PD-aligned layer:
  - `fear`: `+0.6098`
  - `surprise`: `+0.8782`
  - `sadness`: `+0.4820`
  - `happiness`: `+0.2496`
  - `disgust`: `+0.0420`
  - `anger`: `+0.0000`

### Interpretation Against Research Questions
- Q1. Which hidden layers increase self-report target probability most reliably?
  - answer:
    - strong and emotion-specific best layers clearly exist
    - the best layer is not universal across emotions
- Q2. Does the best self-report layer differ from the previously used PD-aligned layer?
  - answer:
    - yes for `5 / 6` emotions
    - only `anger` kept the old PD layer as the best self-report layer
- Q3. Can self-report sweep results guide later PD sweeps?
  - answer:
    - yes
    - but the best self-report alpha is often much higher than what PD generation can safely tolerate
    - therefore next PD work should inherit the new layers but keep the lower PD alpha grid
- Q4. Is coupling likely universal?
  - answer:
    - no
    - current evidence is emotion-specific:
      - `sadness`: strongest coupling candidate
      - `fear`: strong layer-fix candidate because self-report improved dramatically while old PD barely moved
      - `happiness`: dissociation candidate because self-report is strong but prior decision-PD moved negative
      - `disgust`: weak or mixed under current evidence

### Next PD Shortlist
- `sadness @ layer 16`
  - priority: `high`
  - reason: strongest joint evidence for self-report movement and prior positive PD movement
- `surprise @ layer 24`
  - priority: `high`
  - reason: very large self-report gain over the old layer plus prior positive low-intensity PD movement
- `anger @ layer 17`
  - priority: `high`
  - reason: old layer is already correct and prior PD movement is consistently positive at low intensities
- `fear @ layer 15`
  - priority: `medium`
  - reason: prior PD barely moved on the old layer, but self-report says the old layer was simply wrong
- `happiness @ layer 20`
  - priority: `medium`
  - reason: likely dissociation slice worth testing explicitly
- `disgust @ layer 14`
  - priority: `low`
  - reason: self-report gain exists but is small and prior PD movement remains weak

### Locked Design For Next Iteration
- PD generation intensities remain:
  - `1, 2, 4, 6, 8, 10`
- shortlist layers:
  - `anger 17`
  - `sadness 16`
  - `surprise 24`
  - `fear 15`
  - `happiness 20`
  - `disgust 14`

## Iteration 2

### Motivation
- We are still in the exploration phase, not the confirmatory phase.
- The first PD run only tested one layer per emotion, so it is too narrow to answer whether some emotions should be negatively correlated with `p_defect` or whether the previous layer was simply wrong.
- Since Qwen2.5-0.5B fits comfortably on one GPU, the right next move is to expand the PD layer range and run multiple single-GPU jobs in parallel.

### Revised Hypothesis
- Some emotions may legitimately reduce `p_defect`, so the PD exploration should not assume positive coupling.
- `happiness` should be treated as a possible dissociation or negative-correlation case.
- `fear` may have looked flat only because the old PD layer was poor.
- A wider layer sweep around the self-report top candidates will tell us whether PD behavior changes are layer-sensitive in the same way as self-report.

### Changed Factors That May Impact Interpretation
- PD layer range is expanded from one layer per emotion to:
  - `top5` self-report layers per emotion
  - plus the previous PD layer if it is not already included
- PD intensity range is expanded from:
  - `1, 2, 4, 6, 8, 10`
  - to `1, 2, 4, 6, 8, 10, 15, 20`
- Only the two usable PD benchmarks are kept:
  - `game_theory_decision`
  - `game_theory_completion_option_id`
- Execution is parallelized across `4` GPUs with one single-GPU run per worker.

### Implementation
- Added expanded PD sweep runner:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_pd_expanded_multigpu_sweep.py`
- Added tests for:
  - expanded layer-map construction
  - condition-grid construction
  - GPU assignment
  - config writing

### Commands
```bash
conda run -n llm python -m pytest \
  tests/test_pd_expanded_multigpu_sweep.py \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_selfreport_general_sweep_compare.py \
  tests/test_pd_selfreport_coupling_runner.py \
  tests/test_pd_selfreport_coupling_analysis.py \
  tests/test_emotion_experiment_engine_import_safe.py

conda run -n llm python \
  auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_pd_expanded_multigpu_sweep.py \
  --prepare-only

conda run -n llm python \
  auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_pd_expanded_multigpu_sweep.py \
  --skip-existing
```

### Current Run Status
- status:
  - completed
- current expanded PD sweep size:
  - `224` conditions
- GPU plan:
  - `4` GPUs
  - round-robin assignment
  - about `56` condition configs per GPU
- live check at launch:
  - all `4` GPUs showed the model loaded
  - result directories started appearing under:
    - `results/auto_experiments/pd_selfreport_general_sweep_qwen25_05b/pd_expanded_multigpu/`

### Completion Notes
- strict completeness check:
  - expected benchmark outputs: `448`
  - final missing count: `0`
- one run had to be repaired and rerun:
  - config: `surprise_layer_2_intensity_15p0_gpu2.yaml`
  - root cause: `detailed_results.csv` save failed on special characters in responses
  - fix: escape CSV output in `emotion_experiment_engine/experiment.py`
  - rerun completed successfully after the fix

### Added Analysis
- added expanded PD aggregation script:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analyze_expanded_pd_sweep.py`
- analysis outputs:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/pd_expanded_condition_summary.csv`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/pd_expanded_selfreport_joined.csv`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/pd_expanded_layer_summary.csv`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis/pd_expanded_emotion_summary.csv`

### Commands
```bash
conda run -n llm python -m pytest \
  tests/test_pd_expanded_analysis.py \
  tests/test_pd_expanded_multigpu_sweep.py \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_selfreport_general_sweep_compare.py \
  tests/test_pd_selfreport_coupling_runner.py \
  tests/test_pd_selfreport_coupling_analysis.py \
  tests/test_emotion_experiment_engine_import_safe.py

conda run -n llm python \
  auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analyze_expanded_pd_sweep.py
```

### Main Results
- parse-failure gate remained active:
  - all interpretation below uses `unknown_ratio <= 0.10`
- the expanded PD sweep changes the picture materially:
  - we now have valid PD points for all six emotions on at least one layer
  - but the best PD layer is often **not** the best self-report layer
- `game_theory_decision`:
  - strongest positive same-layer candidate:
    - `surprise @ layer 24`
    - `best_valid_delta_defect = +0.046667` at intensity `20`
    - `spearman(delta_p_target, delta_defect) = +0.880952`
  - strongest positive cross-layer candidates:
    - `anger @ layer 18`: `+0.133333`
    - `fear @ layer 16`: `+0.063333`
    - `sadness @ layer 15`: `+0.076667`
  - strongest negative same-layer candidates:
    - `fear @ layer 15`: `-0.126667`
    - `happiness @ layer 20`: `-0.036667`
    - `disgust @ layer 14`: `-0.023333`
- `game_theory_completion_option_id`:
  - strongest positive same-layer candidates:
    - `disgust @ layer 14`: `+0.083333` at intensity `20`
    - `surprise @ layer 24`: `+0.023333` at intensity `20`
  - strongest positive cross-layer candidates:
    - `anger @ layer 18`: `+0.093333`
    - `fear @ layer 16`: `+0.070000`
    - `sadness @ layer 19`: `+0.040000`
  - strongest negative same-layer candidates:
    - `fear @ layer 15`: `-0.080000`
    - `disgust @ layer 14`: `-0.060000`
    - `happiness @ layer 20`: `-0.023333`

### Interpretation Against The Research Question
- Research question:
  - can self-report emotion steering changes be accompanied by external PD behavior changes, and is there a stable coupling relation?
- Current answer:
  - **yes, but not as a universal same-layer positive coupling law**
  - there are at least three distinct regimes in the current evidence:
    - same-layer positive coupling:
      - clearest case is `surprise @ layer 24` on `game_theory_decision`
    - same-layer negative coupling:
      - `fear @ layer 15`
      - `happiness @ layer 20`
      - `disgust @ layer 14` on several slices
    - cross-layer dissociation:
      - the layer that best increases self-report target probability is often not the PD layer with the strongest positive `delta_defect`
- Therefore the current evidence supports:
  - emotion-specific and layer-specific coupling
  - possible negative correlation for some emotions
  - the need to choose PD layers from PD behavior data, not from self-report alone

### Practical Layer/Intensity Takeaways
- if the goal is to test **same-layer coupling**:
  - strongest current candidate is `surprise @ layer 24`, intensity band `10-20`
- if the goal is to test **same-layer negative coupling**:
  - `fear @ layer 15`, intensity band around `8-10`
  - `happiness @ layer 20`, broad negative trend up to `20`
- if the goal is to maximize `p_defect` regardless of whether it uses the best self-report layer:
  - `anger @ layer 18`
  - `fear @ layer 16`
  - `sadness @ layer 15` or `19`

### Hypothesis Revision
- The original hope that self-report-selected layers would directly transfer to PD behavior is too strong.
- A better working hypothesis is:
  - self-report sweep is useful as a **local layer prior**
  - but PD behavior can peak on a nearby layer with different sign structure
  - so the real object of study should be the local layer neighborhood around the self-report optimum, not only the exact optimum itself

## Iteration 3

### Motivation
- The `0.5B` results already show same-layer positive coupling, same-layer negative coupling, and nearby-layer dissociation.
- That means the next scientific question is no longer about a single small model.
- We need to check whether this local layer-structure pattern persists in larger Qwen2.5 models when each model uses its **own** emotion readers.

### Revised Hypothesis
- For `Qwen2.5-1.5B-Instruct` and `Qwen2.5-3B-Instruct`, the best self-report layer will remain emotion-specific.
- The PD-positive layer will often lie near the self-report peak, but not always exactly on it.
- Same-layer negative coupling should persist for at least some emotions, rather than being a `0.5B` artifact.

### Changed Factors That May Impact Interpretation
- Emotion readers will now be generated from each model itself instead of reusing `0.5B`-aligned assumptions.
- PD follow-up will use **local neighborhoods around the best self-report layer** instead of the old `top5 + previous PD layer` expansion.
- Self-report remains full-layer logprob sweep, while PD remains lower-intensity generation with the parse gate.

### Implementation
- generalized self-report sweep runner so it can target model-specific output roots:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_general_selfreport_sweep.py`
- added a model-specific PD neighborhood runner:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_pd_neighborhood_multigpu_sweep.py`
- added tests for:
  - model-specific self-report output routing
  - neighborhood layer clipping at model bounds
  - model-specific PD config writing

### Commands
```bash
conda run -n llm python -m pytest \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_neighborhood_multigpu_sweep.py \
  tests/test_pd_expanded_analysis.py \
  tests/test_pd_expanded_multigpu_sweep.py \
  tests/test_pd_selfreport_general_sweep_compare.py \
  tests/test_pd_selfreport_coupling_runner.py \
  tests/test_pd_selfreport_coupling_analysis.py \
  tests/test_emotion_experiment_engine_import_safe.py
```

### Plan
1. Run full self-report sweep for `Qwen2.5-1.5B-Instruct`.
2. Aggregate best self-report layers into a model-specific analysis root.
3. Build a local PD neighborhood sweep with radius `2` and PD intensities `1,2,4,6,8,10`.
4. Repeat the same process for `Qwen2.5-3B-Instruct`.
5. Compare `0.5B`, `1.5B`, and `3B` on:
   - best self-report layer
   - same-layer PD sign
   - nearest positive PD layer offset
   - nearest negative PD layer offset

### Current Status
- status:
  - infrastructure complete
  - `1.5B` self-report sweep about to start

## Iteration 4

### Motivation
- The `1.5B` all-layer PD completion pass was unexpectedly slow at the tail.
- Investigation showed two resume-path bugs in the all-layer PD runner:
  - `--skip-existing` treated any non-empty `output_dir` as complete, so partially finished conditions were skipped forever.
  - the scheduler only inspected `pending[0]`, so if the queue head targeted a busy GPU it would stop scanning and leave other GPUs idle.

### Revised Hypothesis
- The remaining `1.5B` gap is mostly infrastructure, not science:
  - partially completed PD conditions are stranded by an overly coarse completion check
  - tail latency is amplified by queue-head GPU blocking
- If both bugs are fixed, the existing `1.5B` resume command should:
  - recover missing benchmark outputs without rerunning already complete conditions
  - keep all `4` GPUs busy on the remaining unique gaps

### Changed Factors That May Impact Interpretation
- No prompt, dataset, steering vector, evaluation rule, or intensity grid changed.
- Only the runner semantics changed:
  - completion detection now requires both PD benchmarks to have `summary_behavior_ratio.csv`
  - scheduling now scans for the next config whose assigned GPU is free
- This changes wall-clock behavior and resume correctness, but not the scientific condition definition.

### Implementation
- Added resume-completeness tests:
  - `tests/test_pd_alllayers_multigpu_sweep.py`
- Fixed the all-layer PD runner:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_pd_alllayers_multigpu_sweep.py`
  - new `condition_is_complete()` checks both:
    - `game_theory_completion_option_id`
    - `game_theory_decision`
  - new `next_schedulable_index()` avoids queue-head GPU starvation

### Commands
```bash
conda run -n llm python -m pytest \
  tests/test_pd_alllayers_multigpu_sweep.py \
  tests/test_pd_expanded_multigpu_sweep.py

conda run -n llm python auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_pd_alllayers_multigpu_sweep.py \
  --model-path /home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct \
  --gpu-ids 0,1,2,3 \
  --skip-existing
```

### Regression Check
- `pytest tests/test_pd_alllayers_multigpu_sweep.py tests/test_pd_expanded_multigpu_sweep.py`
  - `7 passed`
- `mypy auto_experiments/pd_selfreport_general_sweep_qwen25_05b/run_pd_alllayers_multigpu_sweep.py`
  - blocked by missing environment stub package for `yaml` (`types-PyYAML`)
  - no code-local type error was reached

### Debug Evidence
- Before the fix:
  - config files in `pd_alllayers_configs/qwen2p5-1p5b-instruct`: `1764`
  - unique PD condition output dirs: `1008`
  - unique missing conditions: `63`
  - unique missing outputs: `125`
  - observed tail behavior: only one GPU active in the resume stage
- After the completion-check fix:
  - `summary_behavior_ratio.csv` count increased from `1889` to `1919`
  - unique missing conditions dropped from `63` to `50`
- After the scheduler fix and relaunch:
  - active `series_runner` processes observed on GPUs `0,1,2,3`
  - `summary_behavior_ratio.csv` count reached `1929`

### Interpretation Against The Research Question
- This iteration does not change the scientific answer yet.
- It removes two infrastructure artifacts that would otherwise bias the comparison between self-report and PD:
  - false completion of partially missing PD conditions
  - reduced multi-GPU utilization near the tail of the sweep
- Therefore the ongoing `1.5B` all-layer PD fill is now scientifically safer to compare against the existing self-report sweep.

### Current Status
- status:
  - `1.5B` all-layer PD resume relaunched with the fixed runner
  - all `4` GPUs observed active on remaining missing conditions
  - analysis will resume after the `1.5B` gap closes

## Iteration 5

### Motivation
- The all-layer PD fill for `0.5B`, `1.5B`, and `3B` is now complete, so the bottleneck is no longer missing data.
- The next question is not whether PD can move, but how to summarize the structure of that movement:
  - if we optimize for behavior change, should we care about signed `delta_defect` or `|delta_defect|`?
  - do all-layer PD effects align with self-report peak layers?
  - do some models show stable contiguous PD behavior bands?

### Revised Hypothesis
- The strongest PD behavior layers are often not the self-report peak layers.
- If the objective is behavior change magnitude, `|delta_defect|` is the right summary because strong decreases in `p_defect` are also meaningful external behavior changes.
- Larger models, especially `3B`, may show cleaner contiguous PD-sensitive layer bands than `0.5B`.

### Changed Factors That May Impact Interpretation
- No new experiments were run in this iteration.
- The interpretation changed in two important ways:
  - PD effect size is now also summarized with `|delta_defect|`
  - contiguous same-sign PD layer bands are treated as an object of analysis, not only best individual points
- The parse-failure gate remains unchanged:
  - all interpretation below uses `unknown_ratio <= 0.10`

### Commands
```bash
python - <<'PY'
from pathlib import Path
import yaml
for slug, model_name in [
    ('qwen2p5-0p5b-instruct','Qwen2.5-0.5B-Instruct'),
    ('qwen2p5-1p5b-instruct','Qwen2.5-1.5B-Instruct'),
    ('qwen2p5-3b-instruct','Qwen2.5-3B-Instruct'),
]:
    config_dir = Path('auto_experiments/pd_selfreport_general_sweep_qwen25_05b/pd_alllayers_configs') / slug
    by_output = {}
    for path in sorted(config_dir.glob('*.yaml')):
        payload = yaml.safe_load(path.read_text())
        by_output.setdefault(payload['output_dir'], payload)
    missing = []
    for output_dir, payload in sorted(by_output.items()):
        out = Path(output_dir)
        miss = []
        for benchmark in ['game_theory_completion_option_id', 'game_theory_decision']:
            if not list(out.glob(f"{model_name}_{benchmark}_Prisoners_Dilemma_*/summary_behavior_ratio.csv")):
                miss.append(benchmark)
        if miss:
            missing.append((payload['experiment_name'], miss))
    print(slug, len(by_output), len(missing))
PY

python - <<'PY'
from pathlib import Path
import pandas as pd
base = Path('auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis_alllayers')
for slug in ['qwen2p5-0p5b-instruct','qwen2p5-1p5b-instruct','qwen2p5-3b-instruct']:
    df = pd.read_csv(base / slug / 'pd_expanded_selfreport_joined.csv')
    df = df[df['unknown_ratio'] <= 0.10].copy()
    df = df.groupby(['benchmark','emotion','layer_1based','intensity'], as_index=False).agg({
        'delta_defect':'mean',
        'delta_p_target_mean':'mean',
    })
    df['abs_delta_defect'] = df['delta_defect'].abs()
    idx = df.groupby(['benchmark','emotion'])['abs_delta_defect'].idxmax()
    print(slug)
    print(df.loc[idx, ['benchmark','emotion','layer_1based','intensity','delta_defect','abs_delta_defect']])
PY
```

### Completion Status
- all-layer PD completion is now verified for all three models:
  - `0.5B`: `864` unique conditions, `missing_conditions = 0`
  - `1.5B`: `1008` unique conditions, `missing_conditions = 0`
  - `3B`: `1296` unique conditions, `missing_conditions = 0`
- multimodel all-layer analyses are present under:
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis_alllayers/qwen2p5-0p5b-instruct/`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis_alllayers/qwen2p5-1p5b-instruct/`
  - `auto_experiments/pd_selfreport_general_sweep_qwen25_05b/analysis_alllayers/qwen2p5-3b-instruct/`

### Main Results: `|delta_defect|` Perspective
- The scientific picture changes when PD behavior is judged by `|delta_defect|` rather than only positive `delta_defect`.
- Strong decreases in `p_defect` are common and should be treated as strong external behavior change, not as failure.
- Model-level summary:
  - `1.5B` shows the largest average best-case PD movement:
    - `completion`: mean best `|delta_defect| = 0.2406`
    - `decision`: mean best `|delta_defect| = 0.2461`
  - `3B` is weaker than `1.5B` on completion but strong on decision:
    - `completion`: `0.1183`
    - `decision`: `0.2128`
  - `0.5B` is lower but still shows strong localized effects:
    - `completion`: `0.1656`
    - `decision`: `0.1694`
- The best PD behavior layer is often far from the self-report best layer:
  - mean absolute offset from self-report peak layer is about `6` to `12` layers depending on model and benchmark
  - maximum observed offset reaches `22` layers
- Therefore:
  - self-report best layer is not a good direct proxy for PD best layer
  - self-report is better interpreted as a broad prior than as a point estimate

### Strongest `|delta_defect|` Cases
- strongest `game_theory_decision` cases across models:
  - `1.5B`, `disgust @ layer 19, intensity 10`: `+0.3867`
  - `1.5B`, `anger @ layer 17, intensity 8`: `+0.3467`
  - `1.5B`, `surprise @ layer 17, intensity 10`: `+0.2867`
  - `3B`, `surprise @ layer 25, intensity 10`: `-0.2833`
  - `3B`, `happiness @ layer 27, intensity 10`: `-0.2633`
  - `3B`, `fear @ layer 23, intensity 10`: `-0.2433`
  - `0.5B`, `happiness @ layer 15, intensity 10`: `-0.2267`
  - `0.5B`, `surprise @ layer 4, intensity 4`: `-0.2200`
- strongest `game_theory_completion_option_id` cases across models:
  - `1.5B`, `anger @ layer 12, intensity 10`: `+0.3400`
  - `1.5B`, `disgust @ layer 17, intensity 10`: `+0.2600`
  - `1.5B`, `sadness @ layer 17, intensity 10`: `+0.2600`
  - `0.5B`, `fear @ layer 10, intensity 6`: `+0.2233`
  - `0.5B`, `disgust @ layer 8, intensity 10`: `+0.2000`

### Same-Layer Results On Self-Report Peak Layer
- If PD is constrained to the self-report best layer, the behavior effect is much smaller than the all-layer best case.
- strongest same-layer cases:
  - `0.5B`, `decision`, `fear @ layer 15`: `|delta_defect| = 0.1267`
  - `1.5B`, `decision`, `disgust @ layer 20`: `|delta_defect| = 0.1233`
  - `3B`, `decision`, `disgust @ layer 26`: `|delta_defect| = 0.1500`
- model-level comparison:
  - `0.5B`, `decision`:
    - best any-layer mean `|delta_defect| = 0.1694`
    - best self-report-layer mean `|delta_defect| = 0.0572`
  - `1.5B`, `decision`:
    - `0.2461` vs `0.0489`
  - `3B`, `decision`:
    - `0.2128` vs `0.0778`
- Therefore:
  - self-report peak layers can sometimes produce PD effects
  - but they usually miss the strongest behavior-changing layer

### Main Results: Contiguous Same-Sign PD Bands
- PD behavior should not be summarized only by isolated best points.
- Under a strict criterion:
  - a layer is called stable positive if all tested intensities yield `delta_defect > 0`
  - stable negative if all tested intensities yield `delta_defect < 0`
  - contiguous runs of length `>= 2` define PD bands
- `3B` shows the cleanest high-layer bands on `game_theory_decision`:
  - `anger`: stable positive band `25-28`
  - `disgust`: stable positive band `25-27`
  - `fear`: stable positive band `25-27`
  - `happiness`: stable negative band `25-29`
  - `surprise`: stable negative band `25-28`
- `1.5B` also shows clear contiguous bands on `game_theory_decision`:
  - `anger`: stable positive band `16-20`
  - `disgust`: stable positive band `16-21`
  - `fear`: stable positive band `17-21`
  - `happiness`: stable negative bands `18-20` and `23-26`
- `0.5B` has weaker and more fragmented structure:
  - `happiness`: stable negative band `15-19`
  - `surprise`: stable positive band `9-11` and stable negative band `15-17`
  - other emotions have shorter and less consistent runs

### Relationship Between PD Bands And Self-Report
- The relation is real, but not a simple peak-to-peak mapping.
- Three regimes are now visible:
  - band overlap:
    - PD band contains the self-report peak layer
    - examples:
      - `1.5B`, `decision`, `disgust`: positive PD band `16-21`, self-report best layer `20`
      - `3B`, `decision`, `disgust`: positive PD band `25-27`, self-report best layer `26`
      - `3B`, `completion`, `fear`: positive PD band `20-22`, self-report best layer `21`
  - same-band but opposite behavioral sign:
    - self-report increases while `delta_defect` is stably negative
    - examples:
      - `0.5B`, `decision`, `happiness`: negative PD band `15-19`, but band mean self-report movement is clearly positive
      - `3B`, `decision`, `happiness`: negative PD band `25-29`, while band mean self-report movement is also positive
      - `3B`, `completion`, `sadness`: negative PD band `21-24` contains self-report best layer `22`
  - broad relation without peak overlap:
    - PD band does not contain the self-report best layer, but self-report in that band is still weakly positive
    - examples:
      - `1.5B`, `decision`, `anger`
      - `1.5B`, `decision`, `fear`
- Therefore:
  - PD-sensitive structure is often band-level
  - self-report gives a coarse prior, not a precise layer target
  - coupling can be positive or negative depending on emotion and model

### Focused Interpretation For `3B`
- `3B` has the clearest model-specific behavior band:
  - high layers `25-29` dominate the `decision` benchmark
- Within that shared high-layer band:
  - `anger / disgust / fear` tend toward positive `delta_defect`
  - `happiness / surprise` tend toward negative `delta_defect`
- This means:
  - the layer position is partly model-specific
  - the sign of behavior change is still emotion-specific
- `3B` intensity trends on representative strong layers are especially clean:
  - `decision, surprise @ 25`:
    - `-0.0233, -0.0533, -0.1600, -0.2567, -0.2667, -0.2833`
  - `decision, happiness @ 27`:
    - `-0.0267, -0.0467, -0.0733, -0.1400, -0.2233, -0.2633`
  - `decision, disgust @ 27`:
    - `+0.0400, +0.0567, +0.0833, +0.1200, +0.1600, +0.1867`
  - `decision, anger @ 27`:
    - `+0.0333, +0.0567, +0.1267, +0.1400, +0.1600, +0.1933`
- So for `3B`:
  - signed `delta_defect` is not uniformly positively correlated with intensity
  - but `|delta_defect|` is close to monotonically increasing on the strongest layers

### Interpretation Against The Research Question
- Can self-report changes accompany external PD behavior change?
  - yes
- Is the relation a simple same-layer positive law?
  - no
- What is the better working picture now?
  - self-report and PD are related at the level of local or contiguous layer structure
  - the strongest PD effect can be positive or negative
  - for larger models, especially `3B`, a model-specific high-layer behavior band is visible
  - emotion determines how steering acts within that band

### Updated Practical Takeaways
- If the objective is maximum external behavior change, optimize `|delta_defect|`, not only positive `delta_defect`.
- If the objective is same-layer coupling, the current best candidates are:
  - `0.5B`, `fear @ 15`
  - `1.5B`, `disgust @ 20`
  - `3B`, `disgust @ 26`
- If the objective is strongest PD effect regardless of sign:
  - `1.5B` currently gives the largest overall effects
  - `3B` gives the cleanest large-model contiguous behavior band
- If the objective is future band-based study design:
  - start from model-specific PD-sensitive bands rather than self-report best single layers

### Iteration 2026-03-20: Qwen3 Multimodel Protocol Migration And Sanity Gate

#### Research Question
- Can the existing self-report and PD coupling protocol be rerun cleanly on `Qwen3-0.6B`, `Qwen3-1.7B`, and `Qwen3-4B` with model-specific readers/vectors, a shared experiment root, and a stricter low-intensity grid that is appropriate for generation-based PD evaluation?

#### Hypothesis
- If the protocol migration is correct, then:
  - the canonical experiment root can be renamed to `pd_selfreport_pd_coupling_multimodel` without breaking runner/analyzer/test behavior
  - both self-report and PD sanity runs on `Qwen3-0.6B` will complete under the new root
  - the Qwen3 sanity outputs will show non-degenerate metrics while keeping `enable_thinking=false`

#### Changed Factors That May Affect Interpretation
- Experiment root:
  - canonical root renamed from `pd_selfreport_general_sweep_qwen25_05b` to `pd_selfreport_pd_coupling_multimodel`
- Self-report intensity grid:
  - reduced to `1,2,4,6,8`
- PD intensity grid:
  - reduced to `1,2,4,6,8`
- Model scope:
  - default model moved from Qwen2.5 to `Qwen3-0.6B`
- Result layout:
  - self-report default outputs now go to `results/auto_experiments/pd_selfreport_pd_coupling_multimodel/self_report_logprob_multimodel/<model_slug>/`
- Qwen3 inference policy:
  - `enable_thinking=false` in the generated configs

#### Implementation Notes
- Fixed a real bug discovered by the sanity run:
  - `run_general_selfreport_sweep.py` treated `BenchmarkConfig` like a dict when setting `sample_limit`
  - fix: assign `benchmark.sample_limit` directly
- Fixed another protocol bug:
  - the default Qwen3 self-report output root still pointed to legacy `self_report_logprob/`
  - fix: default to `self_report_logprob_multimodel/qwen3-0p6b`
- Added regression coverage for:
  - sanity filters on self-report condition grid
  - benchmark sample-limit mutation on `BenchmarkConfig`
  - renamed experiment root in analyzer/sweep tests

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_expanded_multigpu_sweep.py \
  tests/test_pd_alllayers_multigpu_sweep.py \
  tests/test_pd_neighborhood_multigpu_sweep.py \
  tests/test_pd_expanded_analysis.py \
  tests/test_pd_selfreport_general_sweep_compare.py
```
- Result:
  - `25 passed`

#### Mypy Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
mypy --ignore-missing-imports \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/analyze_expanded_pd_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/compare_with_previous_pd.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_neighborhood_multigpu_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_expanded_multigpu_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_general_selfreport_sweep.py
```
- Result:
  - local script issues introduced in this iteration were cleared
  - mypy still reports many pre-existing repo-wide typing problems in shared dependencies such as `constants.py`, `emotion_experiment_engine/config_loader.py`, and dataset modules

#### Sanity Runs
- Self-report sanity command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
CUDA_VISIBLE_DEVICES=0 \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_general_selfreport_sweep.py \
  --model-path /home/jjl7137/huggingface_models/Qwen/Qwen3-0.6B \
  --output-root results/auto_experiments/pd_selfreport_pd_coupling_multimodel/self_report_logprob_multimodel/qwen3-0p6b \
  --emotions anger \
  --layers 1 \
  --intensities 1 \
  --sample-limit 16
```
- Self-report sanity result:
  - output dir:
    - `results/auto_experiments/pd_selfreport_pd_coupling_multimodel/self_report_logprob_multimodel/qwen3-0p6b/anger_layer_1_intensity_1p0/`
  - key metric:
    - `p_target_mean = 0.001866`
    - `delta_p_target_mean = 0.000225`
    - `delta_margin_mean = 0.017151`

- PD sanity command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py \
  --model-path /home/jjl7137/huggingface_models/Qwen/Qwen3-0.6B \
  --emotions anger \
  --layers 1 \
  --intensities 1 \
  --sample-limit 16 \
  --gpu-ids 0 \
  --batch-size 800
```
- PD sanity result:
  - completion benchmark:
    - output dir:
      - `results/auto_experiments/pd_selfreport_pd_coupling_multimodel/pd_alllayers_multimodel/qwen3-0p6b/anger_layer_1/intensity_1p0/Qwen3-0.6B_game_theory_completion_option_id_Prisoners_Dilemma_20260320_012844/`
    - `anger defect = 0.4375`
    - `neutral defect = 0.4375`
    - `delta_defect = 0.0000`
  - decision benchmark:
    - output dir:
      - `results/auto_experiments/pd_selfreport_pd_coupling_multimodel/pd_alllayers_multimodel/qwen3-0p6b/anger_layer_1/intensity_1p0/Qwen3-0.6B_game_theory_decision_Prisoners_Dilemma_20260320_012918/`
    - `anger defect = 0.3125`
    - `neutral defect = 0.3125`
    - `delta_defect = 0.0000`

#### Check Against Hypothesis
- The migration is functionally correct enough to proceed:
  - renamed root works in runners/analyzers/tests
  - Qwen3 self-report sanity completes
  - Qwen3 PD sanity completes on both benchmarks
- The sanity outcomes are intentionally weak in effect size:
  - one low intensity
  - one layer
  - only 16 samples
  - this is acceptable because the goal of the sanity gate was execution validity, not scientific conclusion

#### Next Plan
- Launch full Qwen3 self-report all-layer sweeps for:
  - `Qwen3-0.6B`
  - `Qwen3-1.7B`
  - `Qwen3-4B`
- Launch full Qwen3 PD all-layer sweeps for the same models with PD intensities limited to `1,2,4,6,8`
- Reuse the same canonical experiment root and compare the resulting self-report and PD layer structure after all runs finish

### Iteration 2026-03-20: Qwen3 Sequential GPU Ownership Protocol

#### Research Question
- Can the full Qwen3 multimodel sweep be executed with a cleaner protocol where each model finishes `self-report` before starting `PD`, while GPU ownership stays fixed per model?

#### Hypothesis
- If execution order is fixed inside each model lane, then:
  - the protocol will better match the intended research workflow
  - `Qwen3-0.6B` can own `GPU 0`
  - `Qwen3-1.7B` can own `GPU 1`
  - `Qwen3-4B` can own `GPU 2,3` with tensor parallelism `2`
  - resume behavior will stay correct because `--skip-existing` still gates both self-report and PD conditions

#### Changed Factors That May Affect Interpretation
- Execution policy:
  - changed from phase-wise launch across models to per-model sequence:
    - `self-report -> PD`
- GPU ownership:
  - fixed mapping:
    - `Qwen3-0.6B -> GPU 0`
    - `Qwen3-1.7B -> GPU 1`
    - `Qwen3-4B -> GPU 2,3`
- Tensor parallel configuration:
  - explicit `tensor_parallel_size` is now passed through both the self-report runner and the PD runner
- Failure recovery:
  - orphaned `VLLM::EngineCore` processes from an earlier failed launch had to be removed before the `Qwen3-4B` lane could restart cleanly

#### Implementation Notes
- Added a dedicated orchestration script:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py`
- Updated `run_general_selfreport_sweep.py`:
  - CLI now accepts `--tensor-parallel-size`
  - loading config writes the explicit tensor parallel size into vLLM launch args
- Updated `run_pd_alllayers_multigpu_sweep.py`:
  - CLI now accepts `--tensor-parallel-size`
  - generated PD series configs persist the explicit tensor parallel size
- Added regression coverage for:
  - model-to-GPU mapping
  - per-model execution plan ordering
  - subprocess command construction
  - tensor-parallel propagation in both self-report and PD config builders

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  tests/test_qwen3_multimodel_sequence_runner.py \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_alllayers_multigpu_sweep.py
```
- Result:
  - `19 passed`

#### Mypy Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
mypy --ignore-missing-imports --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_general_selfreport_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py
```
- Result:
  - command does not provide a clean local signal yet
  - mypy is blocked by many pre-existing repo-wide typing errors in shared modules
  - the same run also reports missing `PyYAML` stubs in this repo environment

#### Launch Commands
- `Qwen3-0.6B`:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  --models qwen3-0p6b \
  --batch-size 800
```
- `Qwen3-1.7B`:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  --models qwen3-1p7b \
  --batch-size 800
```
- `Qwen3-4B`:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  --models qwen3-4b \
  --batch-size 800
```

#### Runtime Notes
- Initial `Qwen3-4B` launch failed before model execution:
  - root cause:
    - stale orphan `VLLM::EngineCore` processes were still occupying `GPU 2` and `GPU 3`
    - vLLM then saw only `8.32 / 23.55 GiB` free per device, which was below the configured startup target
- Recovery:
  - confirmed the stale cores were orphaned with `PPID=1`
  - killed only those orphan processes
  - restarted the `Qwen3-4B` sequence lane without changing the scientific config
- This recovery matters scientifically:
  - we did not silently lower the configured GPU memory target
  - we restored the intended experiment environment instead of mutating the experiment

#### Current Status
- `Qwen3-0.6B`:
  - running
  - current phase: `self-report`
- `Qwen3-1.7B`:
  - running
  - current phase: `self-report`
- `Qwen3-4B`:
  - restarted after orphan-process cleanup
  - current phase: `self-report`

#### Check Against Hypothesis
- The protocol change is implemented correctly at the runner level:
  - sequencing is explicit per model
  - GPU mapping is explicit per model
  - tensor parallel size is explicit for the `Qwen3-4B` lane
- Runtime evidence already rules out one bad explanation:
  - the earlier `Qwen3-4B` failure was not caused by the new sequence policy itself
  - it was caused by leftover vLLM engine processes from an earlier failed launch

#### Next Plan
- Let all three Qwen3 lanes finish their full self-report sweep first
- Verify each lane transitions into PD automatically after self-report completion
- After all runs finish:
  - aggregate self-report summaries
  - aggregate PD summaries
  - analyze whether the self-report to PD relation changes qualitatively between `0.6B`, `1.7B`, and `4B`

### Iteration 2026-03-20: Runner Throughput Cleanup For Qwen3 Sweep

#### Research Question
- Can we reduce the observed GPU idle gaps without changing experiment meaning, by cutting avoidable Python-side I/O in self-report and by reducing the number of PD launch units?

#### Hypothesis
- The large utility drops are mostly orchestration artifacts rather than model incapacity:
  - self-report is bursty because each condition runs only a small prompt batch and then writes many intermediate CSV files
  - PD is bursty because the runner launches one config per intensity, causing too many repeated experiment setups

#### Changed Factors That May Affect Interpretation
- Self-report artifact policy:
  - default now keeps summary artifacts needed for analysis and resume
  - heavy per-condition intermediate CSV files are disabled by default
- PD launch granularity:
  - changed from one config per `(emotion, layer, intensity)`
  - to one config per `layer` with the full emotion list and the full low-intensity list inside the config
- PD config staging path:
  - grouped configs now write to `pd_alllayers_configs_grouped/`
  - this prevents accidental mixing with legacy per-intensity config files

#### Implementation Notes
- Updated `run_general_selfreport_sweep.py`:
  - added `output_config.save_full_condition_artifacts`
  - default is `False`
  - summary files used by resume and later analysis are still written
  - heavy raw and intermediate per-condition CSV files are skipped by default
- Updated `run_pd_alllayers_multigpu_sweep.py`:
  - added grouping helper that merges all requested emotions and all requested intensities for the same `layer`
  - grouped config output root now uses `pd_alllayers_configs_grouped`
  - each emitted PD config now carries:
    - the full emotion list for that layer
    - the full low-intensity list for that layer

#### Key Evidence
- Runtime inspection before the change:
  - self-report had only `10` prompt items and `7` target options, so each condition issued only `70` requests
  - observed GPU traces showed short compute bursts followed by long zero-util gaps while Python performed per-condition post-processing and file writes
- Expected launch-count reduction for PD:
  - old config count per model: `6 emotions × N layers × 5 intensities`
  - new config count per model: `N layers`
  - this removes repeated per-emotion and per-intensity launch overhead while preserving the same emotion set and intensity set in results

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  tests/test_qwen3_multimodel_sequence_runner.py \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_alllayers_multigpu_sweep.py
```
- Result:
  - `22 passed`

#### Check Against Hypothesis
- The code changes address the two dominant non-scientific idle sources we identified:
  - self-report no longer spends as much time dumping redundant per-condition artifacts
  - PD no longer fragments the same `layer` sweep into many separate config launches across emotions and intensities
- Scientific knobs remain unchanged:
  - same prompts
  - same emotion readers
  - same layer definitions
  - same low-intensity PD grid
  - same evaluation method

#### Next Plan
- Stop the currently running Qwen3 lanes that were launched under the old runner behavior
- Resume the same three Qwen3 lanes with `--skip-existing` so completed results are preserved and the remaining conditions use the throughput-cleaned runners

### Iteration 2026-03-20: Delegate PD Resume To Series Runner

#### Research Question
- When a grouped PD layer config is only partially complete, can the outer Qwen3 orchestrator resume through the series runner's own report mechanism instead of treating the whole config as a fresh run?

#### Hypothesis
- The most reliable resume behavior is:
  - outer runner only short-circuits configs that are fully complete
  - otherwise it should prefer `emotion_experiment_series_runner --resume <report>`
  - and only fall back to `--config <yaml>` when no prior report exists yet

#### Changed Factors That May Affect Interpretation
- Resume control path:
  - changed from outer-runner restart-by-config for all incomplete PD configs
  - to report-based resume when a prior series report exists in the config output directory
- Scientific factors unchanged:
  - prompts
  - model
  - emotions
  - intensities
  - layer definition
  - evaluation method

#### Implementation Notes
- Updated `run_qwen3_multimodel_sequence.py`:
  - added `_find_resume_report(config_path)`
  - added `_build_pd_series_command(config_path)`
  - PD launch now prefers `--resume <latest_report>` when available
  - otherwise it uses `--config <yaml>` for the first run

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  tests/test_qwen3_multimodel_sequence_runner.py \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_alllayers_multigpu_sweep.py
```
- Result:
  - `25 passed`

#### Key Evidence
- The series runner already exposes native report-based resume via:
  - `python -m emotion_experiment_engine.emotion_experiment_series_runner --resume <report>`
- The new orchestration logic now matches that contract instead of bypassing it

#### Next Plan
- Launch the three Qwen3 model lanes again under the updated runner
- Let self-report finish first
- Then let grouped PD layer jobs resume via series reports whenever partial runs already exist

### Iteration 2026-03-20: Relaunch After Orchestrator Abort

#### Research Question
- After the previous session-level abort killed the three Qwen3 sequence runners, can we resume the same Qwen3 protocol without changing any scientific factor and preserve partial grouped-PD progress?

#### Hypothesis
- Yes.
- Because self-report is already complete for all three Qwen3 models and grouped PD now uses report-based resume, relaunching the same sequence runner should:
  - skip finished self-report conditions
  - prepare the same layer-grouped PD configs
  - resume incomplete grouped PD layer runs from their latest series reports
  - avoid rerunning completed layer groups

#### Changed Factors That May Affect Interpretation
- No scientific factor changed:
  - same prompts
  - same models
  - same emotion readers
  - same layer sweep
  - same intensity grids
  - same evaluation method
- Operational factor only:
  - runners are relaunched under the actual available conda environment on this machine: `llm`
  - launch is moved to detached background processes with log files so a terminal-session abort does not kill the experiments again

#### Debug Evidence Before Relaunch
- Qwen3 self-report completion remains complete:
  - `qwen3-0p6b`: `840 / 840`
  - `qwen3-1p7b`: `840 / 840`
  - `qwen3-4b`: `1080 / 1080`
- No active Qwen3 sequence or grouped-PD runner process was found after the abort.
- Existing grouped-PD layer report counts before relaunch:
  - `qwen3-0p6b`: `17 / 28` layer groups with `memory_experiment_report.json`
  - `qwen3-1p7b`: `16 / 28`
  - `qwen3-4b`: `12 / 36`

#### Relaunch Commands
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm

nohup python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  --models qwen3-0p6b \
  --batch-size 800 \
  > results/auto_experiments/pd_selfreport_pd_coupling_multimodel/logs/qwen3-0p6b_sequence_relaunch_20260320.log 2>&1 &

nohup python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  --models qwen3-1p7b \
  --batch-size 800 \
  > results/auto_experiments/pd_selfreport_pd_coupling_multimodel/logs/qwen3-1p7b_sequence_relaunch_20260320.log 2>&1 &

nohup python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  --models qwen3-4b \
  --batch-size 800 \
  > results/auto_experiments/pd_selfreport_pd_coupling_multimodel/logs/qwen3-4b_sequence_relaunch_20260320.log 2>&1 &
```

#### Next Plan
- Verify the detached runners started successfully
- Confirm they pass self-report immediately via `--skip-existing`
- Monitor grouped PD resume progress from the log files and output roots

### Iteration 2026-03-20: Recover Nested Report Resume For Grouped PD

#### Research Question
- After migrating old grouped-PD outputs into the new `layer_N` directories, can the current Qwen3 resume path also recover cases where the only surviving `memory_experiment_report.json` sits below `layer_N/intensity_*` instead of at the `layer_N` root?

#### Hypothesis
- Yes.
- If both the outer Qwen3 orchestrator and the grouped-PD completion check search nested report locations, then:
  - incomplete migrated layer groups with nested reports can resume from the existing report instead of restarting from config
  - complete detection will remain scientifically correct only when the latest report matches the grouped config

#### Changed Factors That May Affect Interpretation
- Resume lookup only:
  - `_find_resume_report()` now falls back to recursive search under `layer_N`
  - grouped PD `condition_is_complete()` now also checks the latest report, including nested reports
  - config/report compatibility is enforced before treating a migrated run as resumable or complete
- Scientific factors unchanged:
  - same prompts
  - same models
  - same emotions
  - same intensity grid
  - same layer grouping
  - same evaluation method

#### Implementation Notes
- Updated [run_qwen3_multimodel_sequence.py](/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py):
  - `_find_resume_report()` now checks root reports first, then nested reports
  - `_build_pd_series_command()` still rejects mismatched reports and falls back to `--config`
- Updated [run_pd_alllayers_multigpu_sweep.py](/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py):
  - added nested latest-report lookup
  - added config/report compatibility guard to `condition_is_complete()`
- Added regression test in [test_qwen3_multimodel_sequence_runner.py](/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/tests/test_qwen3_multimodel_sequence_runner.py):
  - nested `layer_N/intensity_*` report is now accepted for resume discovery

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  tests/test_qwen3_multimodel_sequence_runner.py \
  tests/test_pd_alllayers_multigpu_sweep.py \
  tests/test_pd_selfreport_general_sweep.py -q
```
- Result:
  - `31 passed`

#### Type Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m mypy --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py
```

#### Key Evidence
- Before the fix:
  - migrated partial run like `qwen3-0p6b/layer_2` had nested reports only
  - outer resume lookup returned `None`, so it would restart from config
- After the fix:
  - nested report lookup is covered by regression test
  - completion logic no longer marks mismatched migrated single-emotion reports as complete

#### Next Plan
- Relaunch the pending Qwen3 PD lanes
- Let resume consume nested migrated reports where available
- Continue filling only the genuinely incomplete layer groups

### Iteration 2026-03-20: Skip Complete Self-Report And Restrict Qwen3 PD To Decision

#### Research Question
- Can we prevent the Qwen3 sequence runner from reloading vLLM for self-report when all self-report condition outputs already exist, and can we reduce the grouped PD rerun to `game_theory_decision` only so the remaining fill targets the benchmark we still care about?

#### Hypothesis
- Yes.
- If the sequence runner checks for complete self-report outputs before launching the self-report subprocess, then reruns will no longer die in the self-report teardown path after an all-skip pass.
- If the grouped PD config emits only `game_theory_decision`, then:
  - grouped PD completion becomes cheaper and better aligned with the current research target
  - resume will only fill the missing `decision` benchmark instead of wasting time on `completion_option_id`

#### Changed Factors That May Affect Interpretation
- PD benchmark set changed for the Qwen3 grouped rerun:
  - from `game_theory_completion_option_id` + `game_theory_decision`
  - to `game_theory_decision` only
- Self-report execution policy changed:
  - complete self-report grids are skipped at the sequence-runner level
  - incomplete self-report grids still run with the same config and `--skip-existing`
- Scientific factors unchanged for the retained benchmark:
  - same model
  - same self-report prompt set
  - same readers
  - same layers
  - same intensities
  - same `game_theory_decision` evaluation method

#### Implementation Notes
- Updated [run_qwen3_multimodel_sequence.py](/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py):
  - added `_selfreport_grid_for_model()`
  - added `_selfreport_is_complete()`
  - `run_model_sequence()` now skips the self-report subprocess when all condition directories already contain:
    - `target_option_softmax_by_steer.csv`
    - `run_metadata.json`
- Updated [run_pd_alllayers_multigpu_sweep.py](/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py):
  - `PD_BENCHMARKS = ["game_theory_decision"]`
  - grouped series configs now emit only `game_theory_decision`

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  auto_experiments/tests/test_qwen3_multimodel_sequence_runner.py \
  auto_experiments/tests/test_pd_alllayers_multigpu_sweep.py \
  auto_experiments/tests/test_pd_selfreport_general_sweep.py -q
```
- Result:
  - `34 passed`

#### Type Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m mypy --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py
```
- Result:
  - blocked by large pre-existing repo-wide mypy failures and missing stubs outside this feature slice
  - no isolated code-local conclusion was possible without changing global typing setup

#### Key Evidence
- Before the change:
  - `qwen3-0p6b` rerun aborted in `run_general_selfreport_sweep.py` after an all-skip pass
  - grouped PD still required both `completion_option_id` and `decision`
- After the change:
  - self-report-complete runs are short-circuited before subprocess launch
  - grouped PD completion and resume logic now target `game_theory_decision` only

#### Next Plan
- Relaunch the three Qwen3 model lanes under the new skip/decision-only logic
- Monitor the transition directly into PD for models whose self-report grids are already complete

### Iteration 2026-03-20: Backfill Legacy Grouped PD Outputs Into `layer_N` Format

#### Research Question
- Can we reorganize the already generated Qwen3 grouped-PD outputs from the old directory naming (`anger_layer_N`) into the current grouped-layer naming (`layer_N`) so resume logic can skip them directly?

#### Hypothesis
- Yes.
- The existing Qwen3 grouped-PD outputs under `anger_layer_N` are legacy naming artifacts of the same grouped-by-layer experiment.
- If we copy missing files from each legacy directory into the matching `layer_N` directory before resume, then:
  - `condition_is_complete()` can evaluate the current config output path directly
  - `emotion_experiment_series_runner --resume` can use the migrated `layer_N` report path
  - future resumes no longer need to depend on the old directory names

#### Changed Factors That May Affect Interpretation
- No scientific factor changed:
  - prompts
  - models
  - emotions
  - intensity grid
  - evaluation rules
  - steering config
- Only result-layout handling changed:
  - added a legacy-output sync step that copies missing files from `*_layer_N` into `layer_N`
  - existing files in `layer_N` are preserved and not overwritten

#### Implementation Notes
- Added `sync_legacy_grouped_outputs()` to:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py`
- The helper:
  - finds legacy sibling directories matching `*_layer_N`
  - copies only missing files into the current `layer_N` directory
  - leaves existing files untouched
- Hooked the sync into:
  - `run_pd_alllayers_multigpu_sweep.py` after config generation
  - `run_qwen3_multimodel_sequence.py` after grouped config preparation

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  tests/test_qwen3_multimodel_sequence_runner.py \
  tests/test_pd_selfreport_general_sweep.py \
  tests/test_pd_alllayers_multigpu_sweep.py
```
- Result:
  - `28 passed`

#### Type Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
mypy --ignore-missing-imports --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_pd_alllayers_multigpu_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_qwen3_multimodel_sequence.py
```
- Result:
  - blocked by repository-wide existing mypy errors outside this change
  - no isolated code-local regression signal was obtained from mypy because the repo is not currently type-clean

#### Real Migration Command
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && python - <<'PY'
from auto_experiments.pd_selfreport_pd_coupling_multimodel import run_pd_alllayers_multigpu_sweep as m
for model in [
    '/home/jjl7137/huggingface_models/Qwen/Qwen3-0.6B',
    '/home/jjl7137/huggingface_models/Qwen/Qwen3-1.7B',
    '/home/jjl7137/huggingface_models/Qwen/Qwen3-4B',
]:
    config_paths = sorted(m.default_config_root(model).glob('*.yaml'))
    synced = m.sync_legacy_grouped_outputs(config_paths)
    print(m.model_slug(model), len(synced))
PY
```

#### Key Evidence
- Actual sync counts:
  - `qwen3-0p6b`: `18` config outputs backfilled
  - `qwen3-1p7b`: `16`
  - `qwen3-4b`: `12`
- After backfill, complete current-format grouped configs:
  - `qwen3-0p6b`: `17 / 28`
  - `qwen3-1p7b`: `9 / 28`
  - `qwen3-4b`: `12 / 36`
- This means the current resume path can now skip all those already-complete `layer_N` configs directly without depending on the legacy directory names.

#### Next Plan
- Resume only the remaining incomplete `layer_N` grouped configs for Qwen3
- Use the new current-format output paths as the sole resume target

### Iteration 2026-03-20: Queue Post-PD Sequential-Game Sweeps Behind Current Qwen3 Runs

#### Research Question
- After the current Qwen3 Prisoner's Dilemma sweep finishes, can we automatically continue into wider behavioral probes on sequential games so emotion-linked behavior change is tested beyond PD?

#### Hypothesis
- Yes.
- A thin post-PD watchdog plus a task-generic all-layer runner is enough.
- We do not need to touch the currently running PD sequence code.
- If we keep the same emotion grid and low generation intensities, then `Escalation_Game` and `Trust_Game_Trustor` can be swept with the same grouped-by-layer strategy used for PD.

#### Changed Factors That May Affect Interpretation
- Behavioral task changed:
  - from `Prisoners_Dilemma`
  - to `Escalation_Game` and `Trust_Game_Trustor`
- Sample limit changed:
  - fixed at `300` per sequential-game task
- No self-report phase is inserted here:
  - this iteration only extends behavior sweeps after the existing PD phase
- Execution policy changed:
  - follow-up tmux sessions are launched only after current `qwen3_*_seq` lanes finish
  - follow-up sessions run sequentially to avoid GPU ownership conflicts

#### Implementation Notes
- Added a task-generic sequential-game runner:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_sequential_games_alllayers_multigpu_sweep.py`
- The runner:
  - reuses the grouped-by-layer config shape from the PD runner
  - keeps the behavior sweep intensity grid at `1, 2, 4, 6, 8`
  - keeps the six-emotion sweep
  - writes outputs under:
    - `results/auto_experiments/pd_selfreport_pd_coupling_multimodel/sequential_games_alllayers_multimodel/`
  - currently targets:
    - `Escalation_Game`
    - `Trust_Game_Trustor`
- Added a post-PD watchdog:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_post_pd_behavior_watchdog.py`
- The watchdog:
  - waits for `qwen3_0p6b_seq`, `qwen3_1p7b_seq`, and `qwen3_4b_seq`
  - launches a Qwen2.5 sequential-game tmux session
  - waits for that session to finish
  - then launches a Qwen3 sequential-game tmux session

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  auto_experiments/tests/test_qwen3_multimodel_sequence_runner.py \
  auto_experiments/tests/test_pd_alllayers_multigpu_sweep.py \
  auto_experiments/tests/test_pd_selfreport_general_sweep.py \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py -q
```
- Result:
  - `40 passed`

#### Type Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m mypy --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_sequential_games_alllayers_multigpu_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_post_pd_behavior_watchdog.py \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py
```
- Result:
  - blocked by existing missing `yaml` stubs in this repo environment
  - no code-local typing regression beyond that environment issue was observed

#### Sanity Command
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_sequential_games_alllayers_multigpu_sweep.py \
  --models qwen2p5-0p5b-instruct \
  --games escalation_game \
  --emotions anger \
  --layers 1 \
  --intensities 1 \
  --prepare-only
```

#### Key Evidence
- New targeted orchestration tests:
  - `auto_experiments/tests/test_post_pd_behavior_orchestration.py`
  - verifies sequential-game benchmark specs
  - verifies task-specific completeness checks
  - verifies watchdog waiting semantics
  - verifies tmux launch command construction
- Existing Qwen3 sequence tests still pass, so the new follow-up path did not alter the currently running PD experiment semantics.

#### Next Plan
- Run the `prepare-only` sanity command once.
- Arm a tmux watchdog session that waits for the current Qwen3 PD lanes.
- Let the watchdog auto-launch the post-PD sequential-game sweeps when those lanes finish.

### Iteration 2026-03-20: Fix Post-PD tmux Launch So Sequential-Game Sweeps Actually Start

#### Research Question
- The Qwen3 PD phase already finished, but why did the queued `Escalation_Game` and `Trust_Game_Trustor` sweeps not stay alive after watchdog launch?

#### Hypothesis
- The watchdog launch command was structurally wrong.
- `tmux new-session` was given a raw shell command string without `bash -lc`, so `source` and `conda activate` were not being executed under a shell that understands them.
- The sessions therefore exited immediately, even though the watchdog log recorded a launch attempt.

#### Changed Factors That May Affect Interpretation
- No scientific factor changed:
  - same models
  - same emotions
  - same intensities
  - same layer sweep
  - same tasks
- Only orchestration changed:
  - tmux launch now wraps the command in `bash -lc`

#### Implementation Notes
- Updated:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_post_pd_behavior_watchdog.py`
- Added a stricter regression check in:
  - `auto_experiments/tests/test_post_pd_behavior_orchestration.py`
- The test now requires:
  - tmux launch command includes `bash`
  - the shell payload is passed via `-lc`

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py \
  auto_experiments/tests/test_qwen3_multimodel_sequence_runner.py \
  auto_experiments/tests/test_pd_alllayers_multigpu_sweep.py -q
```
- Result:
  - `28 passed`

#### Type Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m mypy --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_post_pd_behavior_watchdog.py \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py
```
- Result:
  - blocked by pre-existing missing `yaml` stubs in this environment

#### Key Evidence
- Before fix:
  - watchdog log showed launch attempts
  - no `qwen25_seq_games` or `qwen3_seq_games` tmux sessions remained alive
  - no sequential-game result root was created
- After fix:
  - the launch command format is validated by test
  - manual rerun can now use the corrected shell form directly

#### Next Plan
- Stop the stale watchdog session.
- Relaunch `qwen25_seq_games` and `qwen3_seq_games` with the corrected tmux wrapper.
- Monitor result roots and logs to confirm the sweeps are now progressing.

### Iteration 2026-03-20: Resume Sequential-Game Sweeps With Dual-GPU Qwen2.5-3B And Per-Model Qwen3 tmux Jobs

#### Research Question
- The remaining sequential-game backlog is dominated by `Qwen2.5-3B-Instruct`.
- Can we resume that backlog on `GPU 2,3` with tensor parallel `2`, and can the queued Qwen3 phase launch as three model-specific tmux jobs on `GPU 0`, `GPU 1`, and `GPU 2,3` once the Qwen2.5 phase finishes?

#### Hypothesis
- Yes.
- `Qwen2.5-3B-Instruct` fits the same dual-GPU layout already used for `Qwen3-4B`, so the sequential-game runner should not keep it pinned to one GPU.
- The watchdog should not launch one coarse `qwen3_seq_games` session.
- It should launch three separate Qwen3 tmux jobs so model-to-GPU ownership is explicit and testable.

#### Changed Factors That May Affect Interpretation
- No scientific factor changed:
  - same tasks
  - same prompt format
  - same six emotions
  - same intensity grid `1, 2, 4, 6, 8`
  - same grouped-by-layer sweep
- Only execution policy changed:
  - `Qwen2.5-3B-Instruct` sequential-game resume now uses `CUDA_VISIBLE_DEVICES=2,3`
  - `tensor_parallel_size=2`
  - follow-up Qwen3 orchestration is split into per-model tmux jobs:
    - `qwen3_0p6b_seq_games -> GPU 0`
    - `qwen3_1p7b_seq_games -> GPU 1`
    - `qwen3_4b_seq_games -> GPU 2,3`

#### Implementation Notes
- Updated:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_sequential_games_alllayers_multigpu_sweep.py`
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_post_pd_behavior_watchdog.py`
- Added stronger regression expectations in:
  - `auto_experiments/tests/test_post_pd_behavior_orchestration.py`
- New orchestration behavior:
  - launch all Qwen2.5 model sessions for the sequential-game phase
  - wait for those model sessions to finish
  - then launch all Qwen3 model sessions
  - wait for the Qwen3 model sessions to finish

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py \
  auto_experiments/tests/test_qwen3_multimodel_sequence_runner.py \
  auto_experiments/tests/test_pd_alllayers_multigpu_sweep.py \
  auto_experiments/tests/test_pd_selfreport_general_sweep.py -q
```
- Result:
  - `42 passed`

#### Type Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m mypy --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_sequential_games_alllayers_multigpu_sweep.py \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/run_post_pd_behavior_watchdog.py \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py
```
- Result:
  - blocked by pre-existing missing `yaml` stubs in this environment

#### Sanity Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/run_sequential_games_alllayers_multigpu_sweep.py \
  --models qwen2p5-3b-instruct \
  --games trust_game_trustor \
  --layers 19 \
  --emotions anger \
  --intensities 1 \
  --prepare-only
```
- Result:
  - runner now reports `cuda_visible_devices=2,3` for `Qwen2.5-3B-Instruct`

#### Key Evidence
- Test coverage now explicitly requires:
  - `Qwen2.5-3B-Instruct -> GPU 2,3 / TP=2`
  - `Qwen3-4B -> GPU 2,3 / TP=2`
  - Qwen3 follow-up launch is split into:
    - `qwen3_0p6b_seq_games`
    - `qwen3_1p7b_seq_games`
    - `qwen3_4b_seq_games`
- This closes the previous blind spot where the watchdog only validated one coarse Qwen3 job string instead of explicit per-model tmux jobs.

#### Next Plan
- Stop the old single-GPU `qwen25_3b_seq_games` tmux session.
- Stop the ad-hoc `qwen3_parallel_wait` session.
- Launch a new watchdog session using the per-model phase orchestration.
- Confirm that:
  - `qwen25_3b_seq_games` resumes on `GPU 2,3`
  - after Qwen2.5 completes, the three Qwen3 sequential-game tmux jobs appear with the expected GPU split

### Iteration 2026-03-20: Fix tmux `bash -lc` Argument Splitting For Per-Model Sequential-Game Jobs

#### Research Question
- After splitting the sequential-game watchdog into per-model Qwen2.5 and Qwen3 tmux jobs, do those tmux launches actually stay alive long enough for phase waiting to work?

#### Hypothesis
- The previous per-model watchdog patch still had one shell-argument bug.
- `launch_job()` passed `"-lc <script>"` as a single argv element instead of separate `"-lc"` and `<script>` elements.
- That made the qwen25 phase sessions terminate immediately, so the watchdog skipped straight to launching Qwen3 instead of really waiting.

#### Changed Factors That May Affect Interpretation
- No scientific factor changed.
- Only process launch mechanics changed:
  - `tmux new-session ... bash -lc <script>` is now passed as three separate argv elements:
    - `bash`
    - `-lc`
    - `<script>`

#### Implementation Notes
- Updated:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_post_pd_behavior_watchdog.py`
- Tightened the tmux launch test in:
  - `auto_experiments/tests/test_post_pd_behavior_orchestration.py`
- The regression test now checks the exact argv tail used for tmux launch.

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py \
  auto_experiments/tests/test_qwen3_multimodel_sequence_runner.py \
  auto_experiments/tests/test_pd_alllayers_multigpu_sweep.py \
  auto_experiments/tests/test_pd_selfreport_general_sweep.py -q
```
- Result:
  - `42 passed`

#### Key Evidence
- Before the fix:
  - the watchdog log showed Qwen3 launch lines immediately after qwen25 launch lines
  - that meant the qwen25 tmux jobs were not remaining alive, so the phase wait was ineffective
- After the fix:
  - launch argv shape is now explicitly tested as:
    - `bash`
    - `-lc`
    - `<script>`

#### Next Plan
- Relaunch the watchdog with the corrected tmux argv shape.
- Verify that:
  - `qwen25_3b_seq_games` remains alive on `GPU 2,3`
  - Qwen3 model sessions do not appear until qwen25 model sessions are actually done

### Iteration 2026-03-20: Block Resume Into Old Single-GPU Reports When Tensor Parallel Size Changes

#### Research Question
- Even after relaunching `Qwen2.5-3B-Instruct` with `CUDA_VISIBLE_DEVICES=2,3`, why was the actual vLLM load still showing `tensor_parallel_size=1`?

#### Hypothesis
- The sequential-game runner was still resuming from an old single-GPU report.
- The resume-match logic compared emotions, intensities, layers, and benchmarks, but ignored `loading_config.tensor_parallel_size`.
- As a result, the runner accepted a stale `TP=1` report and resumed into it, defeating the new dual-GPU layout.

#### Changed Factors That May Affect Interpretation
- No scientific factor changed.
- Only resume eligibility changed:
  - if the latest report was created with a different `loading_config.tensor_parallel_size`
  - the runner now rejects `--resume`
  - and falls back to `--config`

#### Implementation Notes
- Updated:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/run_sequential_games_alllayers_multigpu_sweep.py`
- Added regression coverage in:
  - `auto_experiments/tests/test_post_pd_behavior_orchestration.py`
- The new test requires:
  - a config with `tensor_parallel_size=2`
  - and a report with `tensor_parallel_size=1`
  - to select `--config` rather than `--resume`

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  auto_experiments/tests/test_post_pd_behavior_orchestration.py \
  auto_experiments/tests/test_qwen3_multimodel_sequence_runner.py \
  auto_experiments/tests/test_pd_alllayers_multigpu_sweep.py \
  auto_experiments/tests/test_pd_selfreport_general_sweep.py -q
```
- Result:
  - `43 passed`

#### Key Evidence
- Before the fix:
  - `qwen25_3b_seq_games` pane showed:
    - `CUDA_VISIBLE_DEVICES=2,3`
    - but vLLM still loaded with `tensor_parallel_size=1`
- Root cause:
  - stale single-GPU resume report was accepted
- After the fix:
  - resume is blocked when the report TP and config TP differ

#### Next Plan
- Relaunch `qwen25_3b_seq_games` so it starts from the corrected `TP=2` config path.
- Keep the watchdog waiting on qwen25 completion.
- Confirm Qwen3 per-model sessions still do not appear before qwen25 is actually done.

### Iteration 2026-04-07: One-Command Qwen2.5 Positive-Margin Table

#### Research Question
- Can we make the final Qwen2.5 positive-margin validation table reproducible from one command instead of relying on an ad hoc aggregation step?

#### Hypothesis
- The existing saved self-report summaries already contain enough information to rebuild the paper-style Qwen2.5 table directly.
- A standalone reporter can merge:
  - legacy `0.5B` results from `self_report_logprob/`
  - multimodel `1.5B` and `3B` results from `self_report_logprob_multimodel/`
- The positive-margin counts should reproduce the previously reported `x/88` fractions when we count layers with positive 7-way margin.

#### Changed Factors That May Affect Interpretation
- No experiment results changed.
- No steering config changed.
- No benchmark changed.
- Only the reproducibility surface changed:
  - the aggregation is now scripted
  - and the command writes both CSV and Markdown outputs

#### Implementation Notes
- Added:
  - `auto_experiments/pd_selfreport_pd_coupling_multimodel/build_qwen25_positive_margin_table.py`
- Added regression coverage:
  - `auto_experiments/tests/test_qwen25_positive_margin_table.py`
- The new reporter:
  - reads `target_option_softmax_by_steer.csv` plus `run_metadata.json`
  - merges the three Qwen2.5 model roots
  - uses `delta_p_target_vs_top_p_non_target_mean > 0` as the 7-way positive-margin criterion
  - writes:
    - `auto_experiments/pd_selfreport_pd_coupling_multimodel/analysis/qwen25_positive_margin_table.csv`
    - `auto_experiments/pd_selfreport_pd_coupling_multimodel/analysis/qwen25_positive_margin_table.md`
  - prints the Markdown table to stdout for direct one-command use

#### Reproduction
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python auto_experiments/pd_selfreport_pd_coupling_multimodel/build_qwen25_positive_margin_table.py
```

#### Regression Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m pytest \
  auto_experiments/tests/test_qwen25_positive_margin_table.py \
  auto_experiments/tests/test_pd_selfreport_general_sweep.py \
  auto_experiments/tests/test_pd_selfreport_general_sweep_compare.py -q
```
- Result:
  - `18 passed`

#### Type Check
- Command:
```bash
source /home/jjl7137/anaconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m mypy --explicit-package-bases \
  auto_experiments/pd_selfreport_pd_coupling_multimodel/build_qwen25_positive_margin_table.py
```
- Result:
  - `Success: no issues found in 1 source file`

#### Key Evidence
- Real-run output reproduces the expected Qwen2.5 aggregate table shape and counts, including:
  - `anger @ intensity 2`: `24/88 (0.27)`
  - `anger @ intensity 80`: `29/88 (0.33)`
  - `happiness @ intensity 20`: `29/88 (0.33)`
  - `fear @ intensity 40`: `7/88 (0.08)`
- This is the same table family that was previously reconstructed manually for the paper rewrite, but now it is reproducible from one command.
