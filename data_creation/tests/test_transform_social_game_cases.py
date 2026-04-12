#!/usr/bin/env python3
# Purpose: validate data_creation/transform_social_game_cases.py CLI behavior, artifact contracts, and resume logic.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _sample_source_row(case_id: str = "sg_case_1", game_id: str = "game_1") -> dict:
    return {
        "id": case_id,
        "episode_type": "BEAUTY_CONTEST",
        "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN",
        "players": ["ITALY", "AUSTRIA", "FRANCE"],
        "phases": ["S1905M"],
        "events": [{"actor": "ITALY", "recipient": "AUSTRIA", "type": "ATTACK", "phase": "S1905M", "details": {"dest": "VEN"}}],
        "labels": {"mechanism_name": "BC2_COOP_RELATIVE", "winners": ["ITALY"]},
        "metrics": {"num_rounds": 1, "bids": {"ITALY": 1}},
        "source": {"dataset": "standard_no_press.jsonl", "game_id": game_id, "line_number": 10},
    }


def _sample_transformed_row(case_id: str = "sg_case_1", game_id: str = "game_1") -> dict:
    return {
        "scenario": "Coordinated Adriatic Push",
        "description": (
            "You are a Italy commander. The alliance around the Adriatic is unstable, "
            "and every level of commitment changes both your leverage and your exposure."
        ),
        "participants": [{"name": "Italy"}, {"name": "Austria"}, {"name": "France"}],
        "behavior_choices": {
            "commit_0": "Commit no units to the coordinated push this phase.",
            "commit_1": "Commit one unit to the coordinated push this phase.",
            "commit_2": "Commit two units to the coordinated push this phase.",
            "commit_3": "Commit three units to the coordinated push this phase.",
        },
        "previous_actions": [],
        "game_category": "BC2",
        "game_name": "Beauty_Contest",
        "provenance": {
            "id": case_id,
            "source_game_id": game_id,
        },
    }


@pytest.fixture
def prompt_assets(tmp_path: Path) -> tuple[Path, Path]:
    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "beauty_contest_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"episode_type": "BEAUTY_CONTEST"},
                "output": _sample_transformed_row(),
            }
        ],
    )
    return rubric, fewshot


def test_main_requires_social_game_and_paths(tmp_path: Path):
    # data_creation/transform_social_game_cases.py: validate required CLI arguments.
    from data_creation.transform_social_game_cases import main

    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2


def test_transform_run_writes_success_only_outputs(tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch):
    # data_creation/transform_social_game_cases.py: write only loadable rows to the success dataset.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "beauty_contest_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_source_row("sg_case_1", "game_1"), _sample_source_row("sg_case_2", "game_2")])
    rubric, fewshot = prompt_assets

    transformed_rows = [
        _sample_transformed_row("sg_case_1", "game_1"),
        _sample_transformed_row("sg_case_2", "game_2"),
    ]

    def fake_transform_row(*args, **kwargs):
        source_row = kwargs["source_row"]
        return transformed_rows[0] if source_row["id"] == "sg_case_1" else transformed_rows[1]

    monkeypatch.setattr(module, "transform_source_row", fake_transform_row)

    exit_code = module.main(
        [
            "--social-game",
            "beauty_contest",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--few-shot-path",
            str(fewshot),
            "--rubric-path",
            str(rubric),
        ]
    )

    assert exit_code == 0
    success_rows = json.loads((output_dir / "beauty_contest.success.json").read_text(encoding="utf-8"))
    failure_rows = [json.loads(line) for line in (output_dir / "beauty_contest.failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert len(success_rows) == 2
    assert failure_rows == []
    assert metadata["counts"] == {"total": 2, "success": 2, "failed": 0, "skipped": 0}


def test_transform_run_separates_invalid_rows_into_failure_artifacts(tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch):
    # data_creation/transform_social_game_cases.py: keep invalid rows out of the main success dataset.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "beauty_contest_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_source_row("sg_case_1", "game_1"), {"source": {"game_id": "missing_id"}}])
    rubric, fewshot = prompt_assets

    monkeypatch.setattr(module, "transform_source_row", lambda *args, **kwargs: _sample_transformed_row("sg_case_1", "game_1"))

    exit_code = module.main(
        [
            "--social-game",
            "beauty_contest",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--few-shot-path",
            str(fewshot),
            "--rubric-path",
            str(rubric),
        ]
    )

    assert exit_code == 0
    success_rows = json.loads((output_dir / "beauty_contest.success.json").read_text(encoding="utf-8"))
    failure_rows = [json.loads(line) for line in (output_dir / "beauty_contest.failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(success_rows) == 1
    assert len(failure_rows) == 1
    assert failure_rows[0]["stage"] == "input_validation"


def test_main_rejects_unsupported_social_game(tmp_path: Path, prompt_assets: tuple[Path, Path]):
    # data_creation/transform_social_game_cases.py: unsupported social games must fail loudly.
    from data_creation.transform_social_game_cases import main

    input_path = tmp_path / "beauty_contest_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_source_row()])
    rubric, fewshot = prompt_assets

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--social-game",
                "unknown_game",
                "--input-path",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--few-shot-path",
                str(fewshot),
                "--rubric-path",
                str(rubric),
            ]
        )

    assert excinfo.value.code == 2


def test_build_system_prompt_includes_rubric_and_fewshot(tmp_path: Path, prompt_assets: tuple[Path, Path]):
    # data_creation/transform_social_game_cases.py: compose shared rubric text and game-specific few-shot content.
    from data_creation.transform_social_game_cases import build_system_prompt, load_prompt_pack

    rubric, fewshot = prompt_assets
    prompt_pack = load_prompt_pack(
        social_game="beauty_contest",
        rubric_path=rubric,
        few_shot_path=fewshot,
    )

    prompt = build_system_prompt(prompt_pack)

    assert "historical decision-making scenario" in prompt
    assert "Coordinated Adriatic Push" in prompt


def test_resume_skips_completed_successes_without_duplication(tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch):
    # data_creation/transform_social_game_cases.py: use id + source.game_id to skip completed identities.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "beauty_contest_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_source_row("sg_case_1", "game_1"), _sample_source_row("sg_case_2", "game_2")])
    rubric, fewshot = prompt_assets

    _write_json(output_dir / "beauty_contest.success.json", [_sample_transformed_row("sg_case_1", "game_1")])
    (output_dir / "beauty_contest.failures.jsonl").write_text("", encoding="utf-8")
    _write_json(
        output_dir / "run_metadata.json",
        {
            "counts": {"total": 1, "success": 1, "failed": 0, "skipped": 0},
            "completed_identities": ["sg_case_1::game_1"],
        },
    )

    monkeypatch.setattr(module, "transform_source_row", lambda *args, **kwargs: _sample_transformed_row("sg_case_2", "game_2"))

    exit_code = module.main(
        [
            "--social-game",
            "beauty_contest",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--few-shot-path",
            str(fewshot),
            "--rubric-path",
            str(rubric),
        ]
    )

    assert exit_code == 0
    success_rows = json.loads((output_dir / "beauty_contest.success.json").read_text(encoding="utf-8"))
    identities = {(row["provenance"]["id"], row["provenance"]["source_game_id"]) for row in success_rows}
    assert identities == {("sg_case_1", "game_1"), ("sg_case_2", "game_2")}


def test_transform_source_row_uses_game_constructor_as_contract(monkeypatch: pytest.MonkeyPatch, prompt_assets):
    # data_creation/transform_social_game_cases.py: use scenario_class(**data) as the decisive validation boundary.
    from data_creation import transform_social_game_cases as module

    source_row = _sample_source_row("sg_case_1", "game_1")
    rubric, fewshot = prompt_assets
    prompt_pack = module.load_prompt_pack(
        social_game="beauty_contest",
        rubric_path=rubric,
        few_shot_path=fewshot,
    )
    response_payload = _sample_transformed_row("sg_case_1", "game_1")
    response_payload["game_name"] = "Not_The_Canonical_Name"

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(response_payload))
            )
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )

    monkeypatch.setattr(module, "build_openai_client", lambda: fake_client)
    monkeypatch.setattr(
        module,
        "validate_transformed_row",
        lambda row: (_ for _ in ()).throw(AssertionError("local validator should not be called")),
        raising=False,
    )

    transformed = module.transform_source_row(
        source_row=source_row,
        prompt_pack=prompt_pack,
        model_name="deepseek-chat",
    )

    assert transformed["game_name"] == "Not_The_Canonical_Name"
    assert transformed["provenance"]["id"] == "sg_case_1"
