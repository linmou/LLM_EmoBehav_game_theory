# data_creation/tests/test_sealed_auction_job_generator.py
# Purpose: verify job-based sealed auction generator helpers.
import json

import pytest

from data_creation.sealed_auction_job_generator import (
    build_prompt,
    filter_unprocessed_jobs,
    load_existing_records,
    load_fewshot_examples,
    load_persona_jobs,
    normalize_participants,
    parse_model_output,
    read_processed_jobs,
    validate_record,
)


def test_load_persona_jobs_reads_items(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text('{"item": "planner"}\n{"item": "analyst"}\n', encoding="utf-8")

    jobs = load_persona_jobs(path)

    assert jobs == ["planner", "analyst"]


def test_load_fewshot_examples_reads_json(tmp_path):
    path = tmp_path / "fewshot.json"
    path.write_text(
        json.dumps([{"scenario": "S1"}, {"scenario": "S2"}]), encoding="utf-8"
    )

    fewshot = load_fewshot_examples(path)

    assert len(fewshot) == 2
    assert fewshot[0]["scenario"] == "S1"


def test_build_prompt_includes_job_and_fewshot():
    prompt = build_prompt("planner", [{"scenario": "S1"}])

    assert "planner" in prompt
    assert "S1" in prompt
    assert "devote_low" in prompt
    assert "devote_high" in prompt


def test_filter_unprocessed_jobs_skips_processed():
    jobs = ["planner", "analyst", "writer"]
    processed = {"analyst"}

    remaining = filter_unprocessed_jobs(jobs, processed)

    assert remaining == ["planner", "writer"]


def test_read_processed_jobs_from_existing_json(tmp_path):
    output = tmp_path / "out.json"
    output.write_text(
        json.dumps([{"job": "analyst"}, {"job": "writer"}]), encoding="utf-8"
    )

    processed = read_processed_jobs(output)

    assert processed == {"analyst", "writer"}


def test_load_existing_records_missing_file_is_empty(tmp_path):
    output = tmp_path / "missing.json"

    records = load_existing_records(output)

    assert records == []


def test_normalize_participants_produces_four_entries():
    participants = normalize_participants("Tech Leader")

    assert participants[0]["name"] == "You (Tech Leader)"
    assert len(participants) == 4


def test_parse_model_output_accepts_fenced_json():
    raw = (
        "```json\n"
        '{"scenario":"S","description":"D","participants":[{"name":"You (Tech Leader)"},{"name":"Other 1"}],'
        '"behavior_choices":{"devote_low":"L","devote_medium":"M","devote_high":"H"},'
        '"game_category":"SEALED_BID_AUCTION_MULTIPARTY","game_name":"NoReturn_SealedBid_Auction"}'
        "\n```"
    )

    parsed = parse_model_output(raw)

    assert parsed["scenario"] == "S"


def test_validate_record_rejects_missing_behavior_choices():
    record = {
        "scenario": "S1",
        "description": "D1",
        "participants": [{"name": "You (Tech Leader)"}],
    }

    with pytest.raises(ValueError):
        validate_record(record)
