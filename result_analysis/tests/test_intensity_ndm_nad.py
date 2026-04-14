# Responsible for result_analysis/intensity_ndm_nad.py; verifies intensity-wise NDM/NAD aggregation, focal-direction encodings, and LaTeX table rendering.
"""Tests for normalized decision magnitude/alignment analysis over game-theory shuffle results."""

from __future__ import annotations

from pathlib import Path

import pytest

from result_analysis.intensity_ndm_nad import (
    compute_metrics_by_intensity,
    render_latex_table,
    write_latex_table,
)


def _make_record(
    *,
    emotion: str,
    intensity: float,
    item_id: int,
    task_name: str,
    option_id: int,
    options: list[dict[str, object]],
    repeat_id: int = 0,
) -> dict[str, object]:
    def matches_option_id(option: dict[str, object]) -> bool:
        option_id_value = option["id"]
        assert isinstance(option_id_value, (int, float, str))
        return int(option_id_value) == option_id

    chosen = next(opt for opt in options if matches_option_id(opt))
    chosen_text = chosen["text"]
    assert isinstance(chosen_text, str)
    return {
        "emotion": emotion,
        "intensity": intensity,
        "item_id": item_id,
        "task_name": task_name,
        "score": float(option_id),
        "repeat_id": repeat_id,
        "response": {"decision": chosen_text},
        "metadata": {"item_metadata": {"options": options}},
        "error": None,
    }


def test_compute_metrics_by_intensity_uses_focal_direction_and_zero_expected_rows() -> None:
    """result_analysis/intensity_ndm_nad.py: score decisions so positive deltas match the focal human direction, and zero-direction emotions contribute zero to NAD."""
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
            intensity=0.0,
            item_id=0,
            task_name="Prisoners_Dilemma",
            option_id=1,
            options=pd_options,
        ),
        _make_record(
            emotion="happiness",
            intensity=0.8,
            item_id=0,
            task_name="Prisoners_Dilemma",
            option_id=2,
            options=pd_options,
        ),
        _make_record(
            emotion="neutral",
            intensity=0.0,
            item_id=1,
            task_name="Prisoners_Dilemma",
            option_id=2,
            options=pd_options,
        ),
        _make_record(
            emotion="fear",
            intensity=0.8,
            item_id=1,
            task_name="Prisoners_Dilemma",
            option_id=1,
            options=pd_options,
        ),
        _make_record(
            emotion="neutral",
            intensity=0.0,
            item_id=2,
            task_name="Ultimatum_Game_Responder",
            option_id=1,
            options=ug_responder_options,
        ),
        _make_record(
            emotion="anger",
            intensity=0.8,
            item_id=2,
            task_name="Ultimatum_Game_Responder",
            option_id=2,
            options=ug_responder_options,
        ),
    ]

    metrics = compute_metrics_by_intensity(records, model_name="ToyModel")

    row = metrics[(metrics["model"] == "ToyModel") & (metrics["intensity"] == 0.8)].iloc[0]
    assert row["n_pairs"] == 3
    assert row["ndm"] == pytest.approx(1.0, abs=1e-9)
    assert row["nad"] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_render_and_write_latex_table_include_metric_rows(tmp_path: Path) -> None:
    """result_analysis/intensity_ndm_nad.py: emit a compact LaTeX table with NDM and NAD rows for each intensity."""
    table = render_latex_table(
        intensity_values=[0.8, 1.0],
        model_labels={"ToyModel": "Toy"},
        metrics_by_model={
            "ToyModel": {
                0.8: {"ndm": 0.125, "nad": 0.25},
                1.0: {"ndm": 0.5, "nad": -0.125},
            }
        },
    )

    assert "NDM($0.8$)" in table
    assert "NAD($1.0$)" in table
    assert "0.1250" in table
    assert "-0.1250" in table

    out_path = tmp_path / "intensity_impact.tex"
    write_latex_table(out_path, table)

    written = out_path.read_text(encoding="utf-8")
    assert written == table
    assert written.startswith("% Intent:")
