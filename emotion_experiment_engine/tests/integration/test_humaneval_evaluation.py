# Responsible: emotion_experiment_engine/datasets/humaneval.py
# Purpose: Integration test for HumanEval evaluation (multiprocessing sandbox).

import gzip
import json
from pathlib import Path
import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.dataset_factory import create_dataset_from_config

HUMANEVAL_DATA = Path('/home/jjl7137/human-eval/data/HumanEval.jsonl.gz')


@pytest.mark.skipif(not HUMANEVAL_DATA.exists(), reason='HumanEval data file missing')
def test_humaneval_canonical_solution_passes_and_bad_fails_integration(monkeypatch):
    with gzip.open(HUMANEVAL_DATA, 'rt') as fp:
        row = json.loads(next(l for l in fp if l.strip()))

    cfg = BenchmarkConfig(
        name='humaneval',
        task_type='main',
        data_path=HUMANEVAL_DATA,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy='right',
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    ds = create_dataset_from_config(cfg, prompt_wrapper=lambda **kw: kw.get('question',''))
    rec = ds[0]
    gt = rec['ground_truth']

    canonical = gt.get('canonical_solution', '')

    from emotion_experiment_engine.datasets.humaneval import _import_humaneval
    _import_humaneval()
    import importlib
    execution_module = importlib.import_module('human_eval.execution')

    def fake_check_correctness(problem, completion, timeout, completion_id=None):
        cleaned = completion.strip()
        passed = cleaned == canonical.strip()
        return {
            'task_id': problem['task_id'],
            'passed': passed,
            'result': 'passed' if passed else 'failed',
            'completion_id': completion_id,
        }

    monkeypatch.setattr(execution_module, 'check_correctness', fake_check_correctness, raising=False)

    ok_score = ds.evaluate_response(canonical, gt, 'main', rec['prompt'])
    assert ok_score == 1.0

    bad = "def %s(*args, **kwargs):\n    raise Exception('nope')\n" % gt['entry_point']
    bad_score = ds.evaluate_response(bad, gt, 'main', rec['prompt'])
    assert bad_score == 0.0


@pytest.mark.skipif(not HUMANEVAL_DATA.exists(), reason='HumanEval data file missing')
def test_humaneval_evaluate_response_strips_markdown_fences(monkeypatch):
    """Ensure fenced completions with ```python blocks still pass evaluation."""

    with gzip.open(HUMANEVAL_DATA, 'rt') as fp:
        row = json.loads(next(l for l in fp if l.strip()))

    cfg = BenchmarkConfig(
        name='humaneval',
        task_type='main',
        data_path=HUMANEVAL_DATA,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy='right',
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    ds = create_dataset_from_config(cfg, prompt_wrapper=lambda **kw: kw.get('question', ''))
    rec = ds[0]
    gt = rec['ground_truth']

    canonical = gt.get('canonical_solution', '')
    fenced_completion = f"```python\n{canonical}\n```"

    from emotion_experiment_engine.datasets.humaneval import _import_humaneval
    _import_humaneval()
    import importlib
    execution_module = importlib.import_module('human_eval.execution')

    def fake_check_correctness(problem, completion, timeout, completion_id=None):
        if completion.strip() != canonical.strip():
            raise AssertionError('completion was not sanitized before evaluation')
        return {
            'task_id': problem['task_id'],
            'passed': True,
            'result': 'passed',
            'completion_id': completion_id,
        }

    monkeypatch.setattr(execution_module, 'check_correctness', fake_check_correctness, raising=False)

    score = ds.evaluate_response(fenced_completion, gt, 'main', rec['prompt'])
    assert score == 1.0
