"""Tests for scripts/expand_escalation_game_dataset.py: GPT-5 expansion CLI and helpers.
Responsible file: scripts/expand_escalation_game_dataset.py
Purpose: Ensure GPT-driven SC2 escalation dataset expansion uses OAI_CONFIG and meets 10x sizing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from api_configs import OAI_CONFIG
from scripts import expand_escalation_game_dataset as expander


def _scenario(idx: int, you_race: str = "Protoss", opp_race: str = "Terran") -> Dict[str, Any]:
    return {
        "description": f"Scenario {idx} between {you_race} and {opp_race}.",
        "you_play_as": you_race,
        "behaviour_decisions": {
            "escalate": [f"Escalate move {idx}a", f"Escalate move {idx}b"],
            "withdraw": [f"Withdraw move {idx}a", f"Withdraw move {idx}b"],
        },
        "players": {
            "player_1": {
                "race": you_race,
                "role": "You",
                "economy": f"{you_race} economy {idx}",
                "army": f"{you_race} army {idx}",
                "advantage": f"{you_race} advantage {idx}",
            },
            "player_2": {
                "race": opp_race,
                "role": "Opponent",
                "economy": f"{opp_race} economy {idx}",
                "army": f"{opp_race} army {idx}",
                "advantage": f"{opp_race} advantage {idx}",
            },
        },
        "options": [
            {"id": 1, "text": f"Escalate choice {idx}", "category": "escalate", "escalation_strength": 2},
            {"id": 2, "text": f"Withdraw choice {idx}", "category": "withdraw", "escalation_strength": -2},
        ],
        "all_options": [
            {"id": 1, "text": f"Escalate choice {idx}", "category": "escalate", "escalation_strength": 2},
            {"id": 2, "text": f"Second escalate choice {idx}", "category": "escalate", "escalation_strength": 1},
            {"id": 3, "text": f"Withdraw choice {idx}", "category": "withdraw", "escalation_strength": -1},
            {"id": 4, "text": f"Second withdraw choice {idx}", "category": "withdraw", "escalation_strength": -2},
        ],
    }


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = type("msg", (), {"content": content})


class _StubChatCompletions:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.last_kwargs: Dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return type("resp", (), {"choices": [_StubChoice(self._payload)]})


class _StubChat:
    def __init__(self, payload: str) -> None:
        self.completions = _StubChatCompletions(payload)


class _StubClient:
    def __init__(self, payload: str) -> None:
        self.chat = _StubChat(payload)


def test_build_openai_client_uses_oai_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    client = expander.build_openai_client(openai_cls=_Recorder)
    assert isinstance(client, _Recorder)
    assert captured.get("base_url") == OAI_CONFIG.get("base_url")
    assert captured.get("api_key") == OAI_CONFIG.get("api_key")


def test_expand_dataset_to_10x_with_stubbed_client(tmp_path: Path) -> None:
    input_path = tmp_path / "escalation_game.json"
    output_path = tmp_path / "escalation_game_expanded.json"
    base_dataset = [_scenario(1, "Protoss", "Terran")]
    input_path.write_text(json.dumps(base_dataset, indent=2))

    generated_dataset: List[Dict[str, Any]] = [
        _scenario(idx, you_race="Terran", opp_race="Zerg") for idx in range(2, 11)
    ]
    payload = json.dumps({"scenarios": generated_dataset})
    stub_client = _StubClient(payload)

    final_dataset = expander.expand_escalation_game_dataset(
        input_path=input_path,
        output_path=output_path,
        client=stub_client,
        model="gpt-test",
    )

    assert len(final_dataset) == len(base_dataset) * 10
    assert output_path.exists()
    assert json.loads(output_path.read_text()) == final_dataset
    assert stub_client.chat.completions.last_kwargs["model"] == "gpt-test"
    assert stub_client.chat.completions.last_kwargs["response_format"] == {"type": "json_object"}


class _ParallelStubCompletions:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.call_count = 0
        self.last_kwargs: Dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_kwargs = kwargs
        scenarios = [
            _scenario(self.call_count * 100 + i, you_race="Zerg", opp_race="Protoss")
            for i in range(1, self.batch_size + 1)
        ]
        payload = json.dumps({"scenarios": scenarios})
        return type("resp", (), {"choices": [_StubChoice(payload)]})


class _ParallelStubChat:
    def __init__(self, batch_size: int) -> None:
        self.completions = _ParallelStubCompletions(batch_size)


class _ParallelStubClient:
    def __init__(self, batch_size: int) -> None:
        self.chat = _ParallelStubChat(batch_size)


def test_parallel_expansion_batches_requests(tmp_path: Path) -> None:
    input_path = tmp_path / "escalation_game.json"
    output_path = tmp_path / "escalation_game_expanded.json"
    base_dataset = [_scenario(1, "Protoss", "Terran")]
    input_path.write_text(json.dumps(base_dataset, indent=2))

    stub_client = _ParallelStubClient(batch_size=3)

    final_dataset = expander.expand_escalation_game_dataset(
        input_path=input_path,
        output_path=output_path,
        client=stub_client,
        model="gpt-test",
        concurrency=2,
        batch_size=3,
    )

    assert len(final_dataset) == len(base_dataset) * 10
    assert output_path.exists()
    assert stub_client.chat.completions.call_count == 3  # ceil((10-1)/3)
    assert stub_client.chat.completions.last_kwargs["model"] == "gpt-test"
    assert stub_client.chat.completions.last_kwargs["response_format"] == {"type": "json_object"}


def test_parse_generated_scenarios_invalid_json_raises() -> None:
    with pytest.raises(ValueError, match="Invalid JSON from model"):
        expander._parse_generated_scenarios("not valid json")


class _FlakyStubCompletions:
    def __init__(self, valid_batch_size: int) -> None:
        self.valid_batch_size = valid_batch_size
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        if self.call_count == 1:
            payload = ""  # Invalid JSON to trigger retry
        else:
            scenarios = [
                _scenario(1000 + i, you_race="Terran", opp_race="Protoss")
                for i in range(1, self.valid_batch_size + 1)
            ]
            payload = json.dumps({"scenarios": scenarios})
        return type("resp", (), {"choices": [_StubChoice(payload)]})


class _FlakyStubChat:
    def __init__(self, valid_batch_size: int) -> None:
        self.completions = _FlakyStubCompletions(valid_batch_size)


class _FlakyStubClient:
    def __init__(self, valid_batch_size: int) -> None:
        self.chat = _FlakyStubChat(valid_batch_size)


def test_expand_retries_on_invalid_json(tmp_path: Path) -> None:
    input_path = tmp_path / "escalation_game.json"
    output_path = tmp_path / "escalation_game_expanded.json"
    base_dataset = [_scenario(1, "Protoss", "Terran")]
    input_path.write_text(json.dumps(base_dataset, indent=2))

    stub_client = _FlakyStubClient(valid_batch_size=9)

    final_dataset = expander.expand_escalation_game_dataset(
        input_path=input_path,
        output_path=output_path,
        client=stub_client,
        model="gpt-test",
        concurrency=1,
        batch_size=9,
        max_retries=2,
    )

    assert len(final_dataset) == len(base_dataset) * 10
    assert stub_client.chat.completions.call_count == 2  # One failure then one success


class _FailThenSuccessCompletions:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        if self.call_count == 1:
            payload = ""  # Force parse failure on first request
        else:
            scenarios = [
                _scenario(2000 + self.call_count * 10 + i, you_race="Zerg", opp_race="Terran")
                for i in range(1, self.batch_size + 1)
            ]
            payload = json.dumps({"scenarios": scenarios})
        return type("resp", (), {"choices": [_StubChoice(payload)]})


class _FailThenSuccessChat:
    def __init__(self, batch_size: int) -> None:
        self.completions = _FailThenSuccessCompletions(batch_size)


class _FailThenSuccessClient:
    def __init__(self, batch_size: int) -> None:
        self.chat = _FailThenSuccessChat(batch_size)


def test_parallel_continues_when_one_future_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "escalation_game.json"
    output_path = tmp_path / "escalation_game_expanded.json"
    base_dataset = [_scenario(1, "Protoss", "Terran")]
    input_path.write_text(json.dumps(base_dataset, indent=2))

    stub_client = _FailThenSuccessClient(batch_size=5)

    final_dataset = expander.expand_escalation_game_dataset(
        input_path=input_path,
        output_path=output_path,
        client=stub_client,
        model="gpt-test",
        concurrency=2,
        batch_size=5,
        max_retries=1,
        max_rounds=5,
    )

    assert len(final_dataset) == len(base_dataset) * 10
    assert stub_client.chat.completions.call_count >= 2


def test_target_total_overrides_default_multiplier(tmp_path: Path) -> None:
    input_path = tmp_path / "escalation_game.json"
    output_path = tmp_path / "escalation_game_expanded.json"
    base_dataset = [_scenario(1, "Protoss", "Terran")]
    input_path.write_text(json.dumps(base_dataset, indent=2))

    # This stub always returns unique scenarios so deduping will not interfere.
    stub_client = _ParallelStubClient(batch_size=2)

    target_total = 7  # 1 base + 6 generated; not equal to 10x, so we validate override.
    final_dataset = expander.expand_escalation_game_dataset(
        input_path=input_path,
        output_path=output_path,
        client=stub_client,
        model="gpt-test",
        concurrency=2,
        batch_size=2,
        target_total=target_total,
        max_rounds=10,
    )

    assert len(final_dataset) == target_total


def test_render_progress_bar_basic() -> None:
    bar = expander._render_progress_bar(current=5, total=10, width=10)
    assert "[#####.....]" in bar
    assert "5/10" in bar
    assert "50%" in bar
