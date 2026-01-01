# data_creation/tests/test_sealed_auction_generator.py
# Purpose: verify sealed auction generator helpers and validation.
import json

import pytest

from data_creation.sealed_auction_generator import (
    build_prompt,
    generate_one,
    load_fewshot_examples,
    parse_model_output,
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
    assert "devote_low" in prompt
    assert "devote_high" in prompt
    assert "game_category" in prompt
    assert "NoReturn_SealedBid_Auction" in prompt


def test_parse_model_output_accepts_fenced_json():
    raw = (
        "```json\n"
        '{"scenario":"S","description":"D","participants":[{"name":"You (Commander of France)"},{"name":"Commander of England"}],"behavior_choices":{"devote_low":"L","devote_medium":"M","devote_high":"H"},"game_category":"SEALED_BID_AUCTION_MULTIPARTY","game_name":"NoReturn_SealedBid_Auction"}'
        "\n```"
    )

    parsed = parse_model_output(raw)

    assert parsed["scenario"] == "S"


def test_validate_record_accepts_valid_record():
    record = {
        "scenario": "S1",
        "description": "D1",
        "participants": [
            {"name": "You (Commander of France)"},
            {"name": "Commander of England"},
        ],
        "behavior_choices": {
            "devote_low": "L",
            "devote_medium": "M",
            "devote_high": "H",
        },
        "game_category": "SEALED_BID_AUCTION_MULTIPARTY",
        "game_name": "NoReturn_SealedBid_Auction",
    }

    validated = validate_record(record)

    assert validated == record


def test_validate_record_rejects_missing_game_name():
    record = {
        "scenario": "S1",
        "description": "D1",
        "participants": [
            {"name": "You (Commander of France)"},
            {"name": "Commander of England"},
        ],
        "behavior_choices": {
            "devote_low": "L",
            "devote_medium": "M",
            "devote_high": "H",
        },
        "game_category": "SEALED_BID_AUCTION_MULTIPARTY",
    }

    with pytest.raises(ValueError):
        validate_record(record)


def test_generate_one_retries_on_invalid_json(monkeypatch):
    calls = {"count": 0}
    valid = (
        '{"scenario":"S","description":"D","participants":[{"name":"You (Commander of France)"},{"name":"Commander of England"}],'
        '"behavior_choices":{"devote_low":"L","devote_medium":"M","devote_high":"H"},'
        '"game_category":"SEALED_BID_AUCTION_MULTIPARTY","game_name":"NoReturn_SealedBid_Auction"}'
    )

    def fake_request_content(client, model, prompt, temperature):
        calls["count"] += 1
        if calls["count"] == 1:
            return "{invalid json"
        return valid

    monkeypatch.setattr(
        "data_creation.sealed_auction_generator.request_content",
        fake_request_content,
    )

    record = generate_one(
        client=None,
        model="gemini-2.5-flash",
        fewshot=[{"scenario": "S1"}],
        temperature=0.2,
        avoid_titles=[],
        seed=1,
    )

    assert record["scenario"] == "S"
