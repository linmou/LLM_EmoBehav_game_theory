# data_creation/tests/test_beauty_contest_generator.py
# Purpose: verify beauty contest generator helpers for prompt construction and skipping.
import json

from data_creation.beauty_contest_generator import (
    build_prompt,
    filter_unprocessed_jobs,
    load_existing_records,
    load_fewshot_examples,
    load_persona_jobs,
    normalize_behavior_choices,
    normalize_participants,
    trim_records,
    read_processed_jobs,
    render_progress,
    parse_model_output,
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
    assert "commit_0" in prompt
    assert "commit_3" in prompt


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


def test_load_existing_records_accepts_jsonl_fallback(tmp_path):
    output = tmp_path / "out.json"
    output.write_text(
        json.dumps({"job": "analyst"}) + "\n" + json.dumps({"job": "writer"}) + "\n",
        encoding="utf-8",
    )

    records = load_existing_records(output)

    assert records == [{"job": "analyst"}, {"job": "writer"}]


def test_normalize_participants_produces_two_name_entries():
    participants = normalize_participants("Tech Leader")

    assert participants == [
        {"name": "You (Tech Leader)"},
        {"name": "10 Other Tech Leader counterparts"},
    ]


def test_normalize_behavior_choices_maps_option_low_medium_high():
    bc = normalize_behavior_choices(
        {"option_low": "L", "option_medium": "M", "option_high": "H"}
    )

    assert set(bc.keys()) == {"commit_0", "commit_1", "commit_2", "commit_3"}
    assert bc["commit_1"] == "L"
    assert bc["commit_2"] == "M"
    assert bc["commit_3"] == "H"


def test_trim_records_keeps_last_n():
    records = [{"job": f"job{i}"} for i in range(10)]

    trimmed = trim_records(records, max_keep=5)

    assert trimmed == [{"job": f"job{i}"} for i in range(5, 10)]


def test_parse_model_output_accepts_json():
    raw = '{"scenario":"S","description":"D","participants":[{"name":"You (Tech Leader)"},{"name":"10 Competing Tech Firms"}],"behavior_choices":{"commit_0":"0","commit_1":"33","commit_2":"66","commit_3":"100"}}'

    parsed = parse_model_output(raw)

    assert parsed["scenario"] == "S"


def test_parse_model_output_accepts_fenced_json():
    raw = '```json\n{"scenario":"S","description":"D","participants":[{"name":"You (Tech Leader)"},{"name":"10 Competing Tech Firms"}],"behavior_choices":{"commit_0":"0","commit_1":"33","commit_2":"66","commit_3":"100"}}\n```'

    parsed = parse_model_output(raw)

    assert parsed["scenario"] == "S"


def test_render_progress_contains_counts_and_percent():
    text = render_progress(done=3, total=10)

    assert "3/10" in text
    assert "%" in text
