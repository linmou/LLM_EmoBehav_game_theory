# Responsible for result_analysis/game_metric_ndm_nad_compare.py; verifies per-(model, game) NDM/NAD aggregation and GT/Dec comparison LaTeX rendering.
"""Tests for per-game normalized decision magnitude/alignment comparison tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from result_analysis.game_metric_ndm_nad_compare import (
    _load_records_from_run_dir,
    compute_metrics_by_game,
    compute_metrics_for_root,
    render_comparison_latex_table,
    write_latex_table,
)


def _make_record(
    *,
    emotion: str,
    item_id: int,
    task_name: str,
    option_id: int,
    options: list[dict[str, object]],
    repeat_id: int = 0,
    intensity: float = 0.8,
) -> dict[str, object]:
    return {
        "emotion": emotion,
        "intensity": intensity,
        "item_id": item_id,
        "task_name": task_name,
        "score": float(option_id),
        "repeat_id": repeat_id,
        "response": {"decision": "unused"},
        "metadata": {"item_metadata": {"options": options}},
        "error": None,
    }


def test_compute_metrics_by_game_normalizes_by_item_range_and_human_direction() -> None:
    """result_analysis/game_metric_ndm_nad_compare.py: aggregate NDM/NAD at the (model, game) level from neutral-paired item decisions."""
    pd_options = [
        {"id": 1, "text": "Defect", "behavior": "defect"},
        {"id": 2, "text": "Cooperate", "behavior": "cooperate"},
    ]
    ug_responder_options = [
        {"id": 1, "text": "Accept", "behavior": "accept"},
        {"id": 2, "text": "Reject", "behavior": "reject"},
    ]

    records = [
        _make_record(
            emotion="neutral",
            item_id=0,
            task_name="Prisoners_Dilemma",
            option_id=1,
            options=pd_options,
        ),
        _make_record(
            emotion="happiness",
            item_id=0,
            task_name="Prisoners_Dilemma",
            option_id=2,
            options=pd_options,
        ),
        _make_record(
            emotion="neutral",
            item_id=1,
            task_name="Prisoners_Dilemma",
            option_id=2,
            options=pd_options,
        ),
        _make_record(
            emotion="anger",
            item_id=1,
            task_name="Prisoners_Dilemma",
            option_id=1,
            options=pd_options,
        ),
        _make_record(
            emotion="neutral",
            item_id=2,
            task_name="Ultimatum_Game_Responder",
            option_id=1,
            options=ug_responder_options,
        ),
        _make_record(
            emotion="anger",
            item_id=2,
            task_name="Ultimatum_Game_Responder",
            option_id=2,
            options=ug_responder_options,
        ),
    ]

    metrics = compute_metrics_by_game(records, model_name="ToyModel")

    pd_row = metrics[(metrics["model"] == "ToyModel") & (metrics["task"] == "Prisoners_Dilemma")].iloc[0]
    assert pd_row["n_pairs"] == 2
    assert pd_row["ndm"] == pytest.approx(1.0, abs=1e-9)
    assert pd_row["nad"] == pytest.approx(1.0, abs=1e-9)

    ug_row = metrics[(metrics["model"] == "ToyModel") & (metrics["task"] == "Ultimatum_Game_Responder")].iloc[0]
    assert ug_row["n_pairs"] == 1
    assert ug_row["ndm"] == pytest.approx(1.0, abs=1e-9)
    assert ug_row["nad"] == pytest.approx(1.0, abs=1e-9)


def test_render_comparison_latex_table_emits_gt_vs_decision_cells(tmp_path: Path) -> None:
    """result_analysis/game_metric_ndm_nad_compare.py: render the example-table structure with NDM on the first line and NAD on the second line."""
    table = render_comparison_latex_table(
        games=["Prisoners_Dilemma", "Ultimatum_Game_Responder"],
        rows=[
            {
                "model": "ToyModel",
                "model_family": "Toy",
                "params": "1B",
                "ndm_by_game": {
                    "Prisoners_Dilemma": {"gt": 0.25, "decision": 0.50},
                    "Ultimatum_Game_Responder": {"gt": 0.75, "decision": 0.10},
                },
                "nad_by_game": {
                    "Prisoners_Dilemma": {"gt": 0.10, "decision": -0.20},
                    "Ultimatum_Game_Responder": {"gt": 0.30, "decision": 0.40},
                },
                "mean_ndm": {"gt": 0.50, "decision": 0.30},
                "mean_nad": {"gt": 0.20, "decision": 0.10},
            }
        ],
    )

    assert "Model & Params" in table
    assert "Toy & 1B &" in table
    assert r"\shortstack{0.250 / 0.500 \\ 0.100 / -0.200}" in table
    assert r"\shortstack{0.750 / 0.100 \\ 0.300 / 0.400}" in table
    assert r"\shortstack{Mean\\ \tiny NDM (GT / Dec) \\ \tiny NAD (GT / Dec)}" in table

    out_path = tmp_path / "game_metric_ndm_nad_compare.tex"
    write_latex_table(out_path, table)

    written = out_path.read_text(encoding="utf-8")
    assert written == table
    assert written.startswith(r"\begin{table*}[t]")


def test_compute_metrics_for_root_skips_incomplete_newer_run_dirs(tmp_path: Path) -> None:
    """result_analysis/game_metric_ndm_nad_compare.py: choose the latest usable run, not the latest empty timestamp shell."""
    options = [
        {"id": 1, "text": "Defect", "behavior": "defect"},
        {"id": 2, "text": "Cooperate", "behavior": "cooperate"},
    ]
    complete_run = tmp_path / "ToyModel_game_theory_Prisoners_Dilemma_20260330_120000"
    complete_run.mkdir()
    newer_incomplete_run = tmp_path / "ToyModel_game_theory_Prisoners_Dilemma_20260331_120000"
    newer_incomplete_run.mkdir()
    (complete_run / "raw_results.json").write_text(
        json.dumps(
            [
                _make_record(
                    emotion="neutral",
                    item_id=0,
                    task_name="Prisoners_Dilemma",
                    option_id=1,
                    options=options,
                ),
                _make_record(
                    emotion="happiness",
                    item_id=0,
                    task_name="Prisoners_Dilemma",
                    option_id=2,
                    options=options,
                ),
            ]
        ),
        encoding="utf-8",
    )

    metrics = compute_metrics_for_root(tmp_path, "_game_theory_")

    row = metrics.iloc[0]
    assert row["model"] == "ToyModel"
    assert row["task"] == "Prisoners_Dilemma"
    assert row["ndm"] == pytest.approx(1.0, abs=1e-9)


def test_load_records_from_run_dir_recovers_first_json_value(tmp_path: Path) -> None:
    """result_analysis/game_metric_ndm_nad_compare.py: recover the first valid top-level JSON array when raw_results.json has trailing garbage."""
    run_dir = tmp_path / "ToyModel_game_theory_Prisoners_Dilemma_20260330_120000"
    run_dir.mkdir()
    options = [
        {"id": 1, "text": "Defect", "behavior": "defect"},
        {"id": 2, "text": "Cooperate", "behavior": "cooperate"},
    ]
    payload = json.dumps(
        [
            _make_record(
                emotion="neutral",
                item_id=0,
                task_name="Prisoners_Dilemma",
                option_id=1,
                options=options,
            )
        ]
    )
    (run_dir / "raw_results.json").write_text(payload + "\n{\"junk\": true}\n", encoding="utf-8")

    records = _load_records_from_run_dir(run_dir)

    assert len(records) == 1
    assert records[0]["emotion"] == "neutral"


def test_compute_metrics_for_root_skips_runs_with_no_scoreable_rows(tmp_path: Path) -> None:
    """result_analysis/game_metric_ndm_nad_compare.py: skip runs whose raw results cannot produce any neutral-paired decision rows."""
    good_run = tmp_path / "ToyModel_game_theory_Prisoners_Dilemma_20260330_120000"
    bad_run = tmp_path / "OtherModel_game_theory_Prisoners_Dilemma_20260330_120000"
    good_run.mkdir()
    bad_run.mkdir()
    options = [
        {"id": 1, "text": "Defect", "behavior": "defect"},
        {"id": 2, "text": "Cooperate", "behavior": "cooperate"},
    ]
    (good_run / "raw_results.json").write_text(
        json.dumps(
            [
                _make_record(
                    emotion="neutral",
                    item_id=0,
                    task_name="Prisoners_Dilemma",
                    option_id=1,
                    options=options,
                ),
                _make_record(
                    emotion="happiness",
                    item_id=0,
                    task_name="Prisoners_Dilemma",
                    option_id=2,
                    options=options,
                ),
            ]
        ),
        encoding="utf-8",
    )
    (bad_run / "raw_results.json").write_text(
        json.dumps(
            [
                {
                    "emotion": "neutral",
                    "item_id": 0,
                    "task_name": "Prisoners_Dilemma",
                    "score": None,
                    "repeat_id": 0,
                    "metadata": {"item_metadata": {"options": options}},
                    "error": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    metrics = compute_metrics_for_root(tmp_path, "_game_theory_")

    assert metrics["model"].tolist() == ["ToyModel"]


def test_compute_metrics_for_root_skips_runs_without_scoreable_records(tmp_path: Path) -> None:
    """result_analysis/game_metric_ndm_nad_compare.py: skip runs whose raw_results cannot produce any scored rows."""
    bad_run = tmp_path / "ToyModel_game_theory_Prisoners_Dilemma_20260330_120000"
    bad_run.mkdir()
    (bad_run / "raw_results.json").write_text(
        json.dumps(
            [
                {
                    "emotion": "anger",
                    "item_id": 0,
                    "repeat_id": 0,
                    "task_name": "Prisoners_Dilemma",
                    "score": 1.0,
                    "metadata": {"item_metadata": {"options": []}},
                    "error": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    good_run = tmp_path / "ToyModel_game_theory_Prisoners_Dilemma_20260329_120000"
    good_run.mkdir()
    options = [
        {"id": 1, "text": "Defect", "behavior": "defect"},
        {"id": 2, "text": "Cooperate", "behavior": "cooperate"},
    ]
    (good_run / "raw_results.json").write_text(
        json.dumps(
            [
                _make_record(
                    emotion="neutral",
                    item_id=0,
                    task_name="Prisoners_Dilemma",
                    option_id=1,
                    options=options,
                ),
                _make_record(
                    emotion="anger",
                    item_id=0,
                    task_name="Prisoners_Dilemma",
                    option_id=2,
                    options=options,
                ),
            ]
        ),
        encoding="utf-8",
    )

    metrics = compute_metrics_for_root(tmp_path, "_game_theory_")

    assert len(metrics) == 1
    assert metrics.iloc[0]["ndm"] == pytest.approx(1.0, abs=1e-9)
