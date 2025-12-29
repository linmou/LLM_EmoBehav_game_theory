# tests/test_generate_sc2_scenarios_from_intent_map_with_gemini.py
# Purpose: Regression tests for `generate_sc2_scenarios_from_intent_map_with_gemini.py` core utilities.

from __future__ import annotations

import json
from pathlib import Path


def test_intent_category_to_scenario_type_mapping() -> None:
    from generate_sc2_scenarios_from_intent_map_with_gemini import scenario_type_for_intent_category

    assert scenario_type_for_intent_category("air") == "AirControl"
    assert scenario_type_for_intent_category("drop") == "Airdrop"
    assert scenario_type_for_intent_category("base") == "BaseRace"
    assert scenario_type_for_intent_category("gold") == "GoldMineralCompetition"


def test_iter_intent_map_records_reads_required_fields(tmp_path: Path) -> None:
    from generate_sc2_scenarios_from_intent_map_with_gemini import iter_intent_map_records

    jsonl = tmp_path / "intent_map.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "map_image": "datasets/some_image.png",
                        "description": "friendly bases 5; enemy bases 3",
                        "intent_category": "air",
                        "extra": 123,
                        "meta": {"replay": "r1", "frame_id_first": 10},
                    }
                ),
                json.dumps(
                    {
                        "map_image": "datasets/another.png",
                        "description": "something else",
                        "intent_category": "rest",
                        "meta": {"replay": "r2", "frame_id_first": 20},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = list(iter_intent_map_records(jsonl))
    assert records[0]["map_image"] == "datasets/some_image.png"
    assert records[0]["description"] == "friendly bases 5; enemy bases 3"
    assert records[0]["intent_category"] == "air"
    assert records[0]["metadata"]["replay"] == "r1"
    assert records[0]["metadata"]["frame_id_first"] == 10

    assert records[1]["map_image"] == "datasets/another.png"
    assert records[1]["description"] == "something else"
    assert records[1]["intent_category"] == "rest"
    assert records[1]["metadata"]["replay"] == "r2"
    assert records[1]["metadata"]["frame_id_first"] == 20


def test_generate_one_scenario_adds_image_path(tmp_path: Path) -> None:
    from generate_sc2_scenarios_from_intent_map_with_gemini import generate_one_scenario

    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModel:
        def generate_content(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return _Resp(
                json.dumps(
                    {
                        "scenario": "Protoss-vs-Protoss_AirControl",
                        "description": "A situation.",
                        "participants": [{"name": "Protoss"}, {"name": "Protoss"}],
                        "behavior_choices": {
                            "devote_none": "n",
                            "devote_low": "l",
                            "devote_high": "h",
                        },
                    }
                )
            )

    instruction = ["Return JSON only."]
    examples = [
        {
            "scenario": "X",
            "description": "Y",
            "participants": [{"name": "A"}, {"name": "B"}],
            "behavior_choices": {
                "devote_none": "n",
                "devote_low": "l",
                "devote_high": "h",
            },
        }
    ]
    record = {
        "map_image": "datasets/intent_slices/foo.png",
        "description": "friendly bases 5; enemy bases 3",
        "intent_category": "air",
        "metadata": {"replay": "r1", "frame_id_first": 10},
    }

    scenario = generate_one_scenario(
        model=FakeModel(),
        instruction=instruction,
        examples=examples,
        record=record,
    )
    assert scenario["image_path"] == "datasets/intent_slices/foo.png"
    assert scenario["intent_category"] == "air"
    assert scenario["metadata"]["replay"] == "r1"
    assert scenario["metadata"]["frame_id_first"] == 10
    assert scenario["scenario"] == "Protoss-vs-Protoss_AirControl"
