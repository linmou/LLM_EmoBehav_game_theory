# Tests for emotion_experiment_engine.evaluate_saved_series wrapper to batch deferred evaluations
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _ensure_hf_stub() -> None:
    import sys
    import types

    hub = sys.modules.get("huggingface_hub")
    if hub is None:
        hub = types.ModuleType("huggingface_hub")
        sys.modules["huggingface_hub"] = hub
    if not hasattr(hub, "HfFileSystem"):
        class _DummyFileSystem:  # pragma: no cover - simple stub
            pass

        hub.HfFileSystem = _DummyFileSystem

    if "huggingface_hub.hf_file_system" not in sys.modules:
        sys.modules["huggingface_hub.hf_file_system"] = types.ModuleType(
            "huggingface_hub.hf_file_system"
        )


_ensure_hf_stub()


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
        if dry_run:
            mock_eval.return_value = MagicMock()
        else:
            def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
                (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
                (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
                return MagicMock()

            mock_eval.side_effect = _fake_eval

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
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        from emotion_experiment_engine import evaluate_saved_series

        evaluate_saved_series.process_report(series_report, dry_run=False)

    updated = pending_dir / "README.md"
    content = updated.read_text(encoding="utf-8")
    assert "Evaluation Deferred" not in content
    assert "Evaluation Completed" in content


def test_evaluate_saved_series_reprocesses_when_continue_false(series_report: Path) -> None:
    pending_dir = series_report.parent / "pending_run"
    evaluated_dir = series_report.parent / "evaluated_run"

    with patch(
        "emotion_experiment_engine.evaluate_saved_series.evaluate_saved_run"
    ) as mock_eval:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        from emotion_experiment_engine import evaluate_saved_series

        evaluate_saved_series.process_report(
            series_report,
            dry_run=False,
            continue_completed=False,
        )

    called_dirs = {call.args[0] for call in mock_eval.call_args_list}
    assert pending_dir.resolve() in called_dirs
    assert evaluated_dir.resolve() in called_dirs
