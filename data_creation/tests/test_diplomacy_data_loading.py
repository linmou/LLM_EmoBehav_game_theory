import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_creation.create_scenario_langgraph import (  # noqa: E402
    ScenarioCreationConfig,
    load_persona_jobs,
    resolve_diplomacy_record,
)


def test_diplomacy_records_loaded_and_cached(tmp_path, monkeypatch):
    """File: data_creation/tests/test_diplomacy_data_loading.py
    Ensures diplomacy JSONL records become persona jobs and are cached for later lookup."""
    # Prepare a small JSONL file with distinct records
    sample_records = [
        {"id": 0, "phase": "S1901M"},
        {"id": 1, "phase": "F1901M"},
        {"id": 2, "phase": "S1902M"},
    ]
    records_file = tmp_path / "dip_records.jsonl"
    records_file.write_text("\n".join(json.dumps(r) for r in sample_records))

    # Work inside tmp_path so we do not drop cache into the repo root
    monkeypatch.chdir(tmp_path)

    config = ScenarioCreationConfig(
        use_diplomacy_graph=True,
        diplomacy_records_file=str(records_file),
        debug_mode=False,
        debug_num_records=0,
        output_dir=str(tmp_path / "out"),
    )

    persona_jobs = load_persona_jobs(config)
    expected_jobs = [f"rec_{i}" for i in range(len(sample_records))]
    assert persona_jobs == expected_jobs
    assert getattr(config, "diplomacy_records", None) == sample_records

    second_record = resolve_diplomacy_record(config, "rec_1")
    assert second_record == sample_records[1]

    # Out-of-range or malformed jobs should return None
    assert resolve_diplomacy_record(config, "rec_999") is None
    assert resolve_diplomacy_record(config, "invalid_job") is None


def test_diplomacy_respects_num_personas(tmp_path, monkeypatch):
    """Ensure --num-personas limits diplomacy rec_* expansion when not in debug."""
    sample_records = [
        {"id": 0, "phase": "S1901M"},
        {"id": 1, "phase": "F1901M"},
        {"id": 2, "phase": "S1902M"},
        {"id": 3, "phase": "F1902M"},
    ]
    records_file = tmp_path / "dip_records.jsonl"
    records_file.write_text("\n".join(json.dumps(r) for r in sample_records))
    monkeypatch.chdir(tmp_path)

    config = ScenarioCreationConfig(
        use_diplomacy_graph=True,
        diplomacy_records_file=str(records_file),
        debug_mode=False,
        num_personas=2,
        output_dir=str(tmp_path / "out"),
    )

    persona_jobs = load_persona_jobs(config)
    assert persona_jobs == ["rec_0", "rec_1"]
    assert getattr(config, "diplomacy_records", None) == sample_records
