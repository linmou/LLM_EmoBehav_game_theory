# data_creation/tests/test_beauty_contest_diplomacy_generator.py
# Purpose: verify diplomacy beauty contest generator helpers and validation.
import json
import sys
import types

import pytest

from data_creation.beauty_contest_diplomacy_generator import (
    build_prompt,
    build_seed,
    extract_powers_from_fewshot,
    load_fewshot_examples,
    parse_model_output,
    resolve_api_key,
    validate_record,
)


def test_load_fewshot_examples_reads_list(tmp_path):
    path = tmp_path / "fewshot.json"
    path.write_text(json.dumps([{"scenario": "S1"}]), encoding="utf-8")

    fewshot = load_fewshot_examples(path)

    assert len(fewshot) == 1
    assert fewshot[0]["scenario"] == "S1"


def test_build_prompt_includes_required_keys():
    prompt = build_prompt([{"scenario": "S1"}], avoid_titles=[])

    assert "S1" in prompt
    assert "commit_0" in prompt
    assert "commit_3" in prompt
    assert "game_category" in prompt
    assert "Beauty_Contest" in prompt


def test_parse_model_output_accepts_fenced_json():
    raw = '```json\n{"scenario":"S","description":"D","participants":[{"name":"You (Commander)"} , {"name":"Commander of X"}],"behavior_choices":{"commit_0":"0","commit_1":"1","commit_2":"2","commit_3":"3"},"game_category":"BC2","game_name":"Beauty_Contest"}\n```'

    parsed = parse_model_output(raw)

    assert parsed["scenario"] == "S"


def test_build_seed_uses_time_and_players():
    seed = build_seed(["Italy", "Austria"], now=1_700_000_000.0)
    seed_again = build_seed(["Austria", "Italy"], now=1_700_000_000.0)

    assert seed == seed_again


def test_extract_powers_from_fewshot_parses_participants():
    fewshot = [
        {
            "participants": [
                {"name": "You (Commander of Italy)"},
                {"name": "Commander of Austria, Russia and other allied countries"},
            ]
        }
    ]

    powers = extract_powers_from_fewshot(fewshot)

    assert sorted(powers) == ["Austria", "Italy", "Russia"]


def test_resolve_api_key_uses_api_configs(monkeypatch):
    stub = types.SimpleNamespace(GEMINI_CONFIG={"api_key": "cfg-key"})
    monkeypatch.setitem(sys.modules, "api_configs", stub)

    api_key = resolve_api_key(None)

    assert api_key == "cfg-key"


def test_validate_record_accepts_valid_record():
    record = {
        "scenario": "S1",
        "description": "D1",
        "participants": [{"name": "You (Commander of Italy)"}, {"name": "Commander of Austria"}],
        "behavior_choices": {
            "commit_0": "0",
            "commit_1": "1",
            "commit_2": "2",
            "commit_3": "3",
        },
        "game_category": "BC2",
        "game_name": "Beauty_Contest",
    }

    validated = validate_record(record)

    assert validated == record


def test_validate_record_rejects_missing_game_name():
    record = {
        "scenario": "S1",
        "description": "D1",
        "participants": [{"name": "You (Commander of Italy)"}, {"name": "Commander of Austria"}],
        "behavior_choices": {
            "commit_0": "0",
            "commit_1": "1",
            "commit_2": "2",
            "commit_3": "3",
        },
        "game_category": "BC2",
    }

    with pytest.raises(ValueError):
        validate_record(record)
