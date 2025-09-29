# Tests for emotion_experiment_engine.evaluate_saved_series wrapper to batch deferred evaluations
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_run_dir(base: Path, name: str, evaluated: bool) -> Path:
    run_dir = base / name
    run_dir.mkdir()
    readme = run_dir / "README.md"
    if evaluated:
        readme.write_text("# Experiment Results Files\n", encoding="utf-8")
        (run_dir / "evaluation_summary.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
    else:
        readme.write_text("# Evaluation Deferred\n", encoding="utf-8")
    (run_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "raw_results.json").write_text("[]", encoding="utf-8")
    return run_dir


@pytest.fixture
def series_report(tmp_path: Path) -> Path:
    evaluated = _make_run_dir(tmp_path, "evaluated_run", True)
    pending = _make_run_dir(tmp_path, "pending_run", False)
    report = tmp_path / "series_report.json"
    payload = {
        "experiments": {
            "exp_a": {"output_dir": str(evaluated)},
            "exp_b": {"output_dir": str(pending)},
        }
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    return report


@pytest.mark.parametrize("dry_run", [True, False])
def test_evaluate_saved_series_filters_pending_runs(series_report: Path, dry_run: bool) -> None:
    report_dir = series_report.parent
    pending_dir = report_dir / "pending_run"

    with patch(
        "emotion_experiment_engine.evaluate_saved_series.evaluate_saved_run"
    ) as mock_eval:
        mock_eval.return_value = MagicMock()

        from emotion_experiment_engine import evaluate_saved_series

        result = evaluate_saved_series.process_report(series_report, dry_run=dry_run)

    assert pending_dir.resolve() in result.pending_dirs
    assert report_dir / "evaluated_run" not in result.pending_dirs

    if dry_run:
        mock_eval.assert_not_called()
    else:
        mock_eval.assert_called_once_with(pending_dir.resolve(), max_workers=8)


def test_evaluate_saved_series_updates_readme(series_report: Path) -> None:
    pending_dir = series_report.parent / "pending_run"

    with patch(
        "emotion_experiment_engine.evaluate_saved_series.evaluate_saved_run"
    ) as mock_eval:
        mock_eval.return_value = MagicMock()

        from emotion_experiment_engine import evaluate_saved_series

        evaluate_saved_series.process_report(series_report, dry_run=False)

    updated = pending_dir / "README.md"
    content = updated.read_text(encoding="utf-8")
    assert "Evaluation Deferred" not in content
    assert "Evaluation Completed" in content
