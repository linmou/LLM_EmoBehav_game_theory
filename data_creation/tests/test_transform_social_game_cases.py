#!/usr/bin/env python3
# Purpose: validate data_creation/transform_social_game_cases.py CLI behavior, artifact contracts, and resume logic.

from __future__ import annotations

import json
import threading
import time
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


def _sample_escalation_transformed_row(case_id: str = "sg_case_1", game_id: str = "game_1") -> dict:
    return {
        "scenario": "Border Canal Water Standoff",
        "description": (
            "You and a neighboring district are deciding whether to keep normal pumping "
            "or intensify extraction from the same canal during a drought."
        ),
        "participants": [{"name": "You"}, {"name": "Neighbor"}],
        "behavior_choices": {
            "escalate": "Pump more water from the shared canal.",
            "withdraw": "Keep to the normal pumping level.",
        },
        "previous_actions": [
            ["Neighbor", "Pump more water from the shared canal."],
        ],
        "previous_actions_length": 1,
        "game_name": "Escalation_Game",
        "provenance": {
            "id": case_id,
            "source_game_id": game_id,
        },
    }


def _sample_prisoners_dilemma_source_row(
    case_id: str = "sg_pd_1",
    game_id: str = "pd_game_1",
) -> dict:
    return {
        "id": case_id,
        "episode_type": "PRISONERS_DILEMMA",
        "variant_name": "TWO_AGENTS_SINGLE_TURN",
        "players": ["ENGLAND", "FRANCE"],
        "phases": ["F1912M"],
        "events": [
            {
                "actor": "ENGLAND",
                "recipient": "FRANCE",
                "type": "ATTACK",
                "phase": "F1912M",
                "details": {"dest": "BEL", "reason": "unit_or_sc"},
            }
        ],
        "labels": {"round_outcome": "CD"},
        "metrics": {"dyad": ["BEL", "MAR"], "num_rounds": 1},
        "source": {"dataset": "standard_no_press.jsonl", "game_id": game_id, "line_number": 77},
    }


def _sample_prisoners_dilemma_transformed_row(
    case_id: str = "sg_pd_1",
    game_id: str = "pd_game_1",
) -> dict:
    return {
        "scenario": "Belgium Corridor Choice",
        "description": (
            "You are the England commander. Belgium has become a tempting opening between you "
            "and the commander of France, but either side can turn that opening into a costly struggle."
        ),
        "participants": [
            {"name": "You", "profile": "Commander of England"},
            {"name": "France", "profile": "Commander of France"},
        ],
        "behavior_choices": {
            "cooperate": "Leave the Belgium corridor outside your main effort this season.",
            "defect": "Leave the Belgium corridor inside your main effort this season.",
        },
        "game_name": "Prisoners_Dilemma",
        "provenance": {
            "id": case_id,
            "source_game_id": game_id,
        },
    }


def _sample_trust_source_row(case_id: str = "sg_trust_1", game_id: str = "trust_game_1") -> dict:
    return {
        "id": case_id,
        "episode_type": "TRUST",
        "variant_name": "TWO_AGENTS_MULTI_TURN",
        "players": ["AUSTRIA", "ENGLAND"],
        "phases": ["S1915M", "S1916M"],
        "events": [
            {
                "actor": "AUSTRIA",
                "recipient": "ENGLAND",
                "type": "SUPPORT_MOVE",
                "phase": "S1915M",
                "details": {"dest": "BER", "supporter_loc": "SIL", "unit_origin": "PRU"},
            },
            {
                "actor": "AUSTRIA",
                "recipient": "ENGLAND",
                "type": "SUPPORT_MOVE",
                "phase": "F1915M",
                "details": {"dest": "BER", "supporter_loc": "SIL", "unit_origin": "PRU"},
            },
            {
                "actor": "ENGLAND",
                "recipient": "AUSTRIA",
                "type": "CONVOY",
                "phase": "S1916M",
                "details": {"dest": "PAR", "fleet_loc": "MAO", "unit_origin": "VIE"},
            },
        ],
        "labels": {"trust_outcome": "RECIPROCATED"},
        "metrics": {"num_investment_events": 2, "num_rounds": 2, "window_movement_phases": 6},
        "source": {"dataset": "standard_no_press.jsonl", "game_id": game_id, "line_number": 12311},
    }


def _sample_trustee_transformed_row(case_id: str = "sg_trust_1", game_id: str = "trust_game_1") -> dict:
    return {
        "scenario": "Berlin Corridor Return Decision",
        "description": (
            "You are England commander. Berlin has become a more favorable point of pressure "
            "for your side, but the central line is still unstable and other powers can exploit "
            "any slackening of force there."
        ),
        "participants": [
            {"name": "Austria", "profile": "Trustor_Power", "role": "Trustor"},
            {"name": "England", "profile": "Trustee_Power", "role": "Trustee"},
        ],
        "trustor_behavior_choices": {
            "trust_none": "commit 0% of Austria's available operational support to England",
            "trust_low": "commit about 30% of Austria's available operational support to England",
            "trust_high": "commit more than 80% of Austria's available operational support to England",
        },
        "trustee_behavior_choices": {
            "return_none": "return 0% of England's resulting operational support to Austria",
            "return_medium": "return about 40-50% of England's resulting operational support to Austria",
            "return_high": "return more than 80% of England's resulting operational support to Austria",
        },
        "previous_actions": [
            ["Austria", "commit more than 80% of Austria's available operational support to England"],
        ],
        "previous_actions_length": 1,
        "game_name": "Trust_Game_Trustee",
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
                "input": {"episode_type": "BEAUTY_CONTEST", "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": _sample_transformed_row(),
            },
            {
                "input": {"episode_type": "BEAUTY_CONTEST", "variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    **_sample_transformed_row(),
                    "description": "A parallel bargaining channel changes how one extra commitment is interpreted.",
                },
            },
            {
                "input": {"episode_type": "BEAUTY_CONTEST", "variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    **_sample_transformed_row(),
                    "description": "A narrow bilateral contest makes every additional commitment more legible.",
                },
            }
        ],
    )
    return rubric, fewshot


def test_main_requires_social_game_and_paths(tmp_path: Path):
    # data_creation/transform_social_game_cases.py: validate the minimal required CLI arguments.
    from data_creation.transform_social_game_cases import main

    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2


def test_default_prompt_asset_paths_follow_diplomacy_transform_samples_layout():
    # data_creation/transform_social_game_cases.py: derive default prompt assets from data_creation/transform_to_natural_lannguage_samples/diplomacy/.
    from data_creation.transform_social_game_cases import default_few_shot_path, DEFAULT_RUBRIC_PATH

    assert str(default_few_shot_path("beauty_contest")).endswith(
        "data_creation/transform_to_natural_lannguage_samples/diplomacy/beauty_contest_few_shot_examples.json"
    )
    assert str(default_few_shot_path("escalation_game")).endswith(
        "data_creation/transform_to_natural_lannguage_samples/diplomacy/escalation_game_few_shot_examples.json"
    )
    assert str(default_few_shot_path("trust_game_trustee")).endswith(
        "data_creation/transform_to_natural_lannguage_samples/diplomacy/trust_game_trustee_few_shot_examples.json"
    )
    assert str(DEFAULT_RUBRIC_PATH).endswith(
        "data_creation/transform_to_natural_lannguage_samples/diplomacy/transform_rubrics.md"
    )


def test_load_prompt_pack_supports_trust_game_trustee_with_direct_target_mapping(
    tmp_path: Path,
):
    # data_creation/transform_social_game_cases.py: support trust_game_trustee directly as a target game-role mapping.
    from data_creation.transform_social_game_cases import load_prompt_pack
    from games.trust_game import TrustGameTrusteeScenario

    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "trust_game_trustee_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"episode_type": "TRUST", "variant_name": "TWO_AGENTS_MULTI_TURN"},
                "output": _sample_trustee_transformed_row(),
            }
        ],
    )

    prompt_pack = load_prompt_pack(
        social_game="trust_game_trustee",
        rubric_path=rubric,
        few_shot_path=fewshot,
    )

    assert prompt_pack["target_game_name"] == "Trust_Game_Trustee"
    assert prompt_pack["scenario_class"] is TrustGameTrusteeScenario


def test_load_prompt_pack_supports_prisoners_dilemma_with_direct_target_mapping(
    tmp_path: Path,
):
    # data_creation/transform_social_game_cases.py: support prisoners_dilemma directly as a target game mapping.
    from data_creation.transform_social_game_cases import load_prompt_pack
    from games.prisoner_delimma import PrisonerDilemmaScenario

    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "prisoners_dilemma_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"episode_type": "PRISONERS_DILEMMA", "variant_name": "TWO_AGENTS_SINGLE_TURN"},
                "output": _sample_prisoners_dilemma_transformed_row(),
            }
        ],
    )

    prompt_pack = load_prompt_pack(
        social_game="prisoners_dilemma",
        rubric_path=rubric,
        few_shot_path=fewshot,
    )

    assert prompt_pack["target_game_name"] == "Prisoners_Dilemma"
    assert prompt_pack["scenario_class"] is PrisonerDilemmaScenario


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


def test_transform_run_uses_default_prompt_assets_when_paths_are_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: default to repo-owned diplomacy prompt assets when CLI paths are omitted.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "beauty_contest_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_source_row("sg_case_1", "game_1")])

    asset_root = tmp_path / "data_creation" / "transform_to_natural_lannguage_samples" / "diplomacy"
    asset_root.mkdir(parents=True)
    rubric = asset_root / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = asset_root / "beauty_contest_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"episode_type": "BEAUTY_CONTEST", "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": _sample_transformed_row(),
            },
            {
                "input": {"episode_type": "BEAUTY_CONTEST", "variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    **_sample_transformed_row(),
                    "description": "A bilateral signal changes the value of one extra commitment.",
                },
            },
            {
                "input": {"episode_type": "BEAUTY_CONTEST", "variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    **_sample_transformed_row(),
                    "description": "A single rival reads every extra commitment as a sharper signal.",
                },
            }
        ],
    )

    monkeypatch.setattr(module, "DEFAULT_RUBRIC_PATH", rubric)
    monkeypatch.setattr(module, "default_few_shot_path", lambda social_game: asset_root / f"{social_game}_few_shot_examples.json")
    monkeypatch.setattr(module, "transform_source_row", lambda *args, **kwargs: _sample_transformed_row("sg_case_1", "game_1"))

    exit_code = module.main(
        [
            "--social-game",
            "beauty_contest",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    success_rows = json.loads((output_dir / "beauty_contest.success.json").read_text(encoding="utf-8"))
    assert len(success_rows) == 1


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


def test_transform_run_writes_success_only_outputs_for_escalation_game(
    tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: write only loadable Escalation Game rows to the success dataset.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_source_row("sg_case_1", "game_1")])
    rubric, fewshot = prompt_assets

    monkeypatch.setattr(
        module,
        "transform_source_row",
        lambda *args, **kwargs: _sample_escalation_transformed_row("sg_case_1", "game_1"),
    )

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
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
    success_rows = json.loads((output_dir / "escalation_game.success.json").read_text(encoding="utf-8"))
    assert len(success_rows) == 1
    assert success_rows[0]["game_name"] == "Escalation_Game"


def test_transform_run_writes_success_only_outputs_for_prisoners_dilemma(
    tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: write only loadable Prisoners' Dilemma rows to the success dataset.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "prisoners_dilemma_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_prisoners_dilemma_source_row("sg_pd_1", "pd_game_1")])
    rubric, _ = prompt_assets
    fewshot = tmp_path / "prisoners_dilemma_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"episode_type": "PRISONERS_DILEMMA", "variant_name": "TWO_AGENTS_SINGLE_TURN"},
                "output": _sample_prisoners_dilemma_transformed_row("sg_pd_1", "pd_game_1"),
            },
            {
                "input": {"episode_type": "PRISONERS_DILEMMA", "variant_name": "TWO_AGENTS_MULTI_TURN"},
                "output": {
                    **_sample_prisoners_dilemma_transformed_row("sg_pd_2", "pd_game_2"),
                    "description": "The same frontier has become a repeated dilemma across consecutive seasons.",
                },
            },
            {
                "input": {"episode_type": "PRISONERS_DILEMMA", "variant_name": "TWO_AGENTS_MULTI_TURN"},
                "output": {
                    **_sample_prisoners_dilemma_transformed_row("sg_pd_3", "pd_game_3"),
                    "description": "The corridor remains useful, but a repeated standoff has made every shift more legible.",
                },
            },
        ],
    )

    monkeypatch.setattr(
        module,
        "transform_source_row",
        lambda *args, **kwargs: _sample_prisoners_dilemma_transformed_row("sg_pd_1", "pd_game_1"),
    )

    exit_code = module.main(
        [
            "--social-game",
            "prisoners_dilemma",
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
    success_rows = json.loads((output_dir / "prisoners_dilemma.success.json").read_text(encoding="utf-8"))
    assert len(success_rows) == 1
    assert success_rows[0]["game_name"] == "Prisoners_Dilemma"


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


def test_load_prompt_pack_supports_escalation_game_with_explicit_mapping(
    tmp_path: Path, prompt_assets: tuple[Path, Path]
):
    # data_creation/transform_social_game_cases.py: support escalation_game through explicit target mapping.
    from data_creation.transform_social_game_cases import load_prompt_pack
    from games.escalation_game import EscalationGameScenario

    rubric, fewshot = prompt_assets
    prompt_pack = load_prompt_pack(
        social_game="escalation_game",
        rubric_path=rubric,
        few_shot_path=fewshot,
    )

    assert prompt_pack["target_game_name"] == "Escalation_Game"
    assert prompt_pack["scenario_class"] is EscalationGameScenario


def test_run_transform_filters_few_shot_examples_to_run_present_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: derive the runtime few-shot pool from the selected file, then drop variants absent from the current run input.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            {
                **_sample_source_row("sg_case_1", "game_1"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN",
            },
            {
                **_sample_source_row("sg_case_2", "game_2"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "ONE_VS_ONE_SINGLE_TURN",
            }
        ],
    )

    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "escalation_game_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": {"description": "same variant"},
            },
            {
                "input": {"variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": {"description": "same variant two"},
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {"description": "present cross one"},
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {"description": "present cross two"},
            },
            {
                "input": {"variant_name": "THREE_AGENT_MULTI_TURN"},
                "output": {"description": "absent variant"},
            },
        ],
    )

    captured_prompt_packs: list[dict] = []

    def fake_transform_candidates(*args, **kwargs):
        captured_prompt_packs.append(kwargs["prompt_pack"])
        return [_sample_escalation_transformed_row("sg_case_1", "game_1")]

    monkeypatch.setattr(module, "transform_source_row_candidates", fake_transform_candidates)

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
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
    assert len(captured_prompt_packs) == 2
    assert {
        example["input"]["variant_name"]
        for prompt_pack in captured_prompt_packs
        for example in prompt_pack["few_shot_examples"]
    } == {"OVER_TWO_AGENTS_SINGLE_TURN", "ONE_VS_ONE_SINGLE_TURN"}


def test_run_transform_fails_when_few_shot_pool_cannot_supply_two_cross_variant_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: fail loudly when a multi-variant run cannot supply the required two cross-variant examples for a row.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            {
                **_sample_source_row("sg_case_1", "game_1"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN",
            },
            {
                **_sample_source_row("sg_case_2", "game_2"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "ONE_VS_ONE_SINGLE_TURN",
            },
        ],
    )

    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "escalation_game_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": {"description": "over-two same one"},
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {"description": "one-vs-one same one"},
            },
        ],
    )
    monkeypatch.setattr(
        module,
        "transform_source_row_candidates",
        lambda *args, **kwargs: [
            _sample_escalation_transformed_row(
                kwargs["source_row"]["id"],
                kwargs["source_row"]["source"]["game_id"],
            )
        ],
    )

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
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
    success_rows = json.loads((output_dir / "escalation_game.success.json").read_text(encoding="utf-8"))
    failure_rows = [json.loads(line) for line in (output_dir / "escalation_game.failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert success_rows == []
    assert len(failure_rows) == 2
    assert {row["stage"] for row in failure_rows} == {"few_shot_selection"}


def test_run_transform_fails_when_few_shot_pool_cannot_supply_same_variant_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: fail loudly when a row's own variant is absent from the eligible few-shot pool.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            {
                **_sample_source_row("sg_case_1", "game_1"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN",
            },
            {
                **_sample_source_row("sg_case_2", "game_2"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "ONE_VS_ONE_SINGLE_TURN",
            },
        ],
    )

    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "escalation_game_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {"description": "one-vs-one same one"},
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {"description": "one-vs-one same two"},
            },
        ],
    )
    monkeypatch.setattr(
        module,
        "transform_source_row_candidates",
        lambda *args, **kwargs: [
            _sample_escalation_transformed_row(
                kwargs["source_row"]["id"],
                kwargs["source_row"]["source"]["game_id"],
            )
        ],
    )

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
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
    failure_rows = [
        json.loads(line)
        for line in (output_dir / "escalation_game.failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert any(
        "same-variant example" in row["message"]
        for row in failure_rows
    )


def test_run_metadata_records_run_variants_for_few_shot_pool_reproducibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: persist the run-present variant set in run_metadata.json so few-shot pool derivation is reproducible.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            {
                **_sample_source_row("sg_case_1", "game_1"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN",
            },
            {
                **_sample_source_row("sg_case_2", "game_2"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "ONE_VS_ONE_SINGLE_TURN",
            },
        ],
    )

    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "escalation_game_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": {"description": "over-two same one"},
            },
            {
                "input": {"variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": {"description": "over-two same two"},
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {"description": "one-vs-one same one"},
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {"description": "one-vs-one same two"},
            },
        ],
    )

    monkeypatch.setattr(
        module,
        "transform_source_row_candidates",
        lambda *args, **kwargs: [
            _sample_escalation_transformed_row(
                kwargs["source_row"]["id"],
                kwargs["source_row"]["source"]["game_id"],
            )
        ],
    )

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
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
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_variants"] == ["ONE_VS_ONE_SINGLE_TURN", "OVER_TWO_AGENTS_SINGLE_TURN"]


def test_run_transform_builds_row_specific_few_shot_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: build a separate same-variant-plus-two-cross-variant few-shot pack for each row.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            {
                **_sample_source_row("sg_case_1", "game_1"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "OVER_TWO_AGENTS_SINGLE_TURN",
            },
            {
                **_sample_source_row("sg_case_2", "game_2"),
                "episode_type": "ESCALATION_GAME",
                "variant_name": "ONE_VS_ONE_SINGLE_TURN",
            },
        ],
    )

    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text(
        "Transform structured data into a historical decision-making scenario.",
        encoding="utf-8",
    )
    fewshot = tmp_path / "escalation_game_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": {"variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": {
                    "description": "over-two same one",
                    "behavior_choices": {"escalate": "a1", "withdraw": "a2"},
                },
            },
            {
                "input": {"variant_name": "OVER_TWO_AGENTS_SINGLE_TURN"},
                "output": {
                    "description": "over-two same two",
                    "behavior_choices": {"escalate": "a3", "withdraw": "a4"},
                },
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    "description": "one-vs-one same one",
                    "behavior_choices": {"escalate": "b1", "withdraw": "b2"},
                },
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    "description": "one-vs-one same two",
                    "behavior_choices": {"escalate": "b3", "withdraw": "b4"},
                },
            },
            {
                "input": {"variant_name": "THREE_AGENT_MULTI_TURN"},
                "output": {
                    "description": "absent variant",
                    "behavior_choices": {"escalate": "c1", "withdraw": "c2"},
                },
            },
        ],
    )

    prompt_variants_by_row: dict[str, list[str]] = {}

    def fake_transform_candidates(*args, **kwargs):
        source_row = kwargs["source_row"]
        prompt_pack = kwargs["prompt_pack"]
        prompt_variants_by_row[source_row["id"]] = [
            example["input"]["variant_name"] for example in prompt_pack["few_shot_examples"]
        ]
        return [
            _sample_escalation_transformed_row(
                source_row["id"],
                source_row["source"]["game_id"],
            )
        ]

    monkeypatch.setattr(module, "transform_source_row_candidates", fake_transform_candidates)

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
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
    assert prompt_variants_by_row["sg_case_1"] == [
        "OVER_TWO_AGENTS_SINGLE_TURN",
        "OVER_TWO_AGENTS_SINGLE_TURN",
        "ONE_VS_ONE_SINGLE_TURN",
        "ONE_VS_ONE_SINGLE_TURN",
    ]
    assert prompt_variants_by_row["sg_case_2"] == [
        "ONE_VS_ONE_SINGLE_TURN",
        "ONE_VS_ONE_SINGLE_TURN",
        "OVER_TWO_AGENTS_SINGLE_TURN",
        "OVER_TWO_AGENTS_SINGLE_TURN",
    ]


def test_rank_examples_by_ngram_gain_uses_behavior_choices_in_lexical_surface():
    # data_creation/transform_social_game_cases.py: rank few-shot examples on description plus behavior_choices, not description alone.
    from data_creation.transform_social_game_cases import rank_examples_by_ngram_gain

    ranked = rank_examples_by_ngram_gain(
        [
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    "description": "shared description",
                    "behavior_choices": {
                        "escalate": "repeat signal",
                        "withdraw": "repeat signal",
                    },
                },
            },
            {
                "input": {"variant_name": "ONE_VS_ONE_SINGLE_TURN"},
                "output": {
                    "description": "shared description",
                    "behavior_choices": {
                        "escalate": "fresh leverage phrase",
                        "withdraw": "fresh de-escalation phrase",
                    },
                },
            },
        ]
    )

    assert ranked[0]["output"]["behavior_choices"]["escalate"] == "fresh leverage phrase"


def test_inject_game_fields_overrides_empty_escalation_payoff_matrix_from_model_output():
    # data_creation/transform_social_game_cases.py: replace model-emitted empty payoff_matrix with runtime game contract.
    from data_creation.transform_social_game_cases import inject_game_fields, social_game_config

    prompt_pack = social_game_config("escalation_game")
    payload = inject_game_fields(
        {
            "scenario": "Border Canal Water Standoff",
            "description": "You and a neighboring district are deciding whether to escalate.",
            "participants": [{"name": "You"}, {"name": "Neighbor"}],
            "behavior_choices": {
                "escalate": "Pump more water from the shared canal.",
                "withdraw": "Keep to the normal pumping level.",
            },
            "previous_actions": [],
            "payoff_matrix": {},
        },
        prompt_pack,
    )

    assert payload["payoff_matrix"] == prompt_pack["payoff_matrix"]


def test_write_json_serializes_pydantic_runtime_payoff_matrix(tmp_path: Path):
    # data_creation/transform_social_game_cases.py: serialize runtime PayoffMatrix objects when writing success artifacts.
    from data_creation.transform_social_game_cases import social_game_config, write_json

    output_path = tmp_path / "artifact.json"
    prompt_pack = social_game_config("escalation_game")

    write_json(
        output_path,
        [
            {
                "scenario": "Border Canal Water Standoff",
                "payoff_matrix": prompt_pack["payoff_matrix"],
            }
        ],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["payoff_matrix"]["payoff_leaves"]


def test_write_json_round_trips_escalation_success_rows_through_saved_artifact(tmp_path: Path):
    # data_creation/transform_social_game_cases.py: save escalation success artifacts in a shape that still reloads via EscalationGameScenario.
    from data_creation.transform_social_game_cases import inject_game_fields, social_game_config, write_json
    from games.escalation_game import EscalationGameScenario

    output_path = tmp_path / "artifact.json"
    expected_row = inject_game_fields(
        _sample_escalation_transformed_row("sg_case_1", "game_1"),
        social_game_config("escalation_game"),
    )

    write_json(output_path, [expected_row])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    EscalationGameScenario(**payload[0])


def test_write_jsonl_serializes_pydantic_runtime_payoff_matrix(tmp_path: Path):
    # data_creation/transform_social_game_cases.py: serialize runtime PayoffMatrix objects when writing candidate JSONL artifacts.
    from data_creation.transform_social_game_cases import social_game_config, write_jsonl

    output_path = tmp_path / "artifact.jsonl"
    prompt_pack = social_game_config("escalation_game")

    write_jsonl(
        output_path,
        [
            {
                "scenario": "Border Canal Water Standoff",
                "payoff_matrix": prompt_pack["payoff_matrix"],
            }
        ],
    )

    payload = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payload[0]["payoff_matrix"]["payoff_leaves"]


def test_social_game_config_derives_runtime_contract_from_game_configs():
    # data_creation/transform_social_game_cases.py: derive runtime contract constants from games/game_configs.py.
    from constants import GameNames
    from data_creation.transform_social_game_cases import social_game_config
    from games.game_configs import get_game_config

    beauty_cfg = social_game_config("beauty_contest")
    escalation_cfg = social_game_config("escalation_game")

    assert beauty_cfg["game_name"] is GameNames.BEAUTY_CONTEST
    assert escalation_cfg["game_name"] is GameNames.ESCALATION_GAME
    assert beauty_cfg["scenario_class"] is get_game_config(GameNames.BEAUTY_CONTEST)["scenario_class"]
    assert escalation_cfg["scenario_class"] is get_game_config(GameNames.ESCALATION_GAME)["scenario_class"]


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


def test_resume_skips_completed_escalation_successes_without_duplication(
    tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: preserve resume behavior for escalation_game artifact paths too.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(input_path, [_sample_source_row("sg_case_1", "game_1"), _sample_source_row("sg_case_2", "game_2")])
    rubric, fewshot = prompt_assets

    _write_json(
        output_dir / "escalation_game.success.json",
        [_sample_escalation_transformed_row("sg_case_1", "game_1")],
    )
    (output_dir / "escalation_game.failures.jsonl").write_text("", encoding="utf-8")
    _write_json(
        output_dir / "run_metadata.json",
        {
            "counts": {"total": 1, "success": 1, "failed": 0, "skipped": 0},
            "completed_identities": ["sg_case_1::game_1"],
        },
    )

    monkeypatch.setattr(
        module,
        "transform_source_row",
        lambda *args, **kwargs: _sample_escalation_transformed_row("sg_case_2", "game_2"),
    )

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
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
    success_rows = json.loads((output_dir / "escalation_game.success.json").read_text(encoding="utf-8"))
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

    assert transformed["game_name"] == "Beauty_Contest"
    assert transformed["provenance"]["id"] == "sg_case_1"


def test_transform_source_row_passes_explicit_request_timeout_to_model_client(
    monkeypatch: pytest.MonkeyPatch,
    prompt_assets,
):
    # data_creation/transform_social_game_cases.py: pass explicit request timeout through to the model client call so stalled reads do not hang forever.
    from data_creation import transform_social_game_cases as module

    source_row = _sample_source_row("sg_case_1", "game_1")
    rubric, fewshot = prompt_assets
    prompt_pack = module.load_prompt_pack(
        social_game="beauty_contest",
        rubric_path=rubric,
        few_shot_path=fewshot,
    )
    response_payload = _sample_transformed_row("sg_case_1", "game_1")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(response_payload))
            )
        ]
    )
    seen_kwargs: dict[str, Any] = {}

    def fake_create(**kwargs):
        seen_kwargs.update(kwargs)
        return response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )

    monkeypatch.setattr(module, "build_openai_client", lambda: fake_client)

    module.transform_source_row(
        source_row=source_row,
        prompt_pack=prompt_pack,
        model_name="deepseek-chat",
        request_timeout_seconds=17.5,
    )

    assert seen_kwargs["timeout"] == 17.5


def test_transform_source_row_canonicalizes_structured_trustee_previous_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: canonicalize trustee previous_actions onto declared behavior choices before scenario validation.
    from data_creation import transform_social_game_cases as module

    source_row = _sample_trust_source_row()
    rubric = tmp_path / "transform_rubrics.md"
    rubric.write_text("Transform structured data into a historical decision-making scenario.", encoding="utf-8")
    fewshot = tmp_path / "trust_game_trustee_few_shot_examples.json"
    _write_json(
        fewshot,
        [
            {
                "input": _sample_trust_source_row(),
                "output": _sample_trustee_transformed_row(),
            }
        ],
    )
    prompt_pack = module.load_prompt_pack(
        social_game="trust_game_trustee",
        rubric_path=rubric,
        few_shot_path=fewshot,
        run_variants={"TWO_AGENTS_MULTI_TURN"},
    )
    response_payload = {
        "scenario": "Berlin Corridor Return Decision",
        "description": (
            "You are England commander. Berlin has become a more favorable point of pressure for your "
            "side, but the central line is still unstable and other powers can exploit any slackening of force there."
        ),
        "participants": [
            {
                "name": "Austria",
                "profile": "Central power that repeatedly invested support",
                "role": "Trustor",
            },
            {
                "name": "England",
                "profile": "Partner power deciding whether to reciprocate",
                "role": "Trustee",
            },
        ],
        "previous_actions": [
            {
                "round": 1,
                "round_summary": "Austria provided support for England's Berlin campaign, and England made a limited return response.",
                "actions": [
                    {
                        "participant": "Austria",
                        "action": "provide a moderate level of operational support for England's advance toward Berlin",
                    },
                    {
                        "participant": "England",
                        "action": "return a limited level of operational support to Austria after benefiting from the campaign",
                    },
                ],
            },
            {
                "round": 2,
                "round_summary": "Austria again provided support for England's Berlin campaign, and England made another return response.",
                "actions": [
                    {
                        "participant": "Austria",
                        "action": "provide a high level of operational support for England's Berlin campaign",
                    },
                    {
                        "participant": "England",
                        "action": "return a moderate level of operational support to Austria after benefiting again from the campaign",
                    },
                ],
            },
        ],
        "trustor_behavior_choices": {
            "trust_none": "commit 0% of Austria's available operational support to England's Berlin campaign",
            "trust_low": "commit about 30% of Austria's available operational support to England's Berlin campaign",
            "trust_high": "commit more than 80% of Austria's available operational support to England's Berlin campaign",
        },
        "trustee_behavior_choices": {
            "return_none": "return 0% of England's resulting operational support to Austria",
            "return_medium": "return about 40-50% of England's resulting operational support to Austria",
            "return_high": "return more than 80% of England's resulting operational support to Austria",
        },
    }

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

    transformed = module.transform_source_row(
        source_row=source_row,
        prompt_pack=prompt_pack,
        model_name="deepseek-chat",
    )

    assert transformed["previous_actions"] == [
        {
            "round": 1,
            "round_summary": "Austria provided support for England's Berlin campaign, and England made a limited return response.",
            "actions": [
                {
                    "participant": "Austria",
                    "action": "commit about 30% of Austria's available operational support to England's Berlin campaign",
                },
                {
                    "participant": "England",
                    "action": "return about 40-50% of England's resulting operational support to Austria",
                },
            ],
        },
        {
            "round": 2,
            "round_summary": "Austria again provided support for England's Berlin campaign, and England made another return response.",
            "actions": [
                {
                    "participant": "Austria",
                    "action": "commit more than 80% of Austria's available operational support to England's Berlin campaign",
                },
                {
                    "participant": "England",
                    "action": "return about 40-50% of England's resulting operational support to Austria",
                },
            ],
        },
    ]


def test_run_transform_uses_multiple_workers_when_requested(
    tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: dispatch independent transforms concurrently when num-workers > 1.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            _sample_source_row("sg_case_1", "game_1"),
            _sample_source_row("sg_case_2", "game_2"),
            _sample_source_row("sg_case_3", "game_3"),
            _sample_source_row("sg_case_4", "game_4"),
        ],
    )
    rubric, fewshot = prompt_assets

    active = 0
    peak_active = 0
    lock = threading.Lock()

    def fake_transform_row(*args, **kwargs):
        nonlocal active, peak_active
        source_row = kwargs["source_row"]
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return _sample_escalation_transformed_row(source_row["id"], source_row["source"]["game_id"])

    monkeypatch.setattr(module, "transform_source_row", fake_transform_row)

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--few-shot-path",
            str(fewshot),
            "--rubric-path",
            str(rubric),
            "--num-workers",
            "4",
        ]
    )

    assert exit_code == 0
    assert peak_active > 1


def test_run_transform_with_num_candidates_selects_lower_overlap_description_and_writes_artifacts(
    tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: generate multiple valid candidates, select lower-overlap descriptions, and write candidate/diversity artifacts.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            _sample_source_row("sg_case_1", "game_1"),
            _sample_source_row("sg_case_2", "game_2"),
        ],
    )
    rubric, fewshot = prompt_assets

    repeated_1 = _sample_escalation_transformed_row("sg_case_1", "game_1")
    repeated_1["description"] = (
        "You are a You commander. The border canal is unstable. "
        "A costly deadlock can follow if both sides keep pressing."
    )
    repeated_2 = _sample_escalation_transformed_row("sg_case_2", "game_2")
    repeated_2["description"] = (
        "You are a You commander. The border canal is unstable. "
        "A costly deadlock can follow if both sides keep pressing."
    )
    diverse_2 = _sample_escalation_transformed_row("sg_case_2", "game_2")
    diverse_2["description"] = (
        "You are a You commander. The shared canal has become a crowded bargaining point, "
        "and stepping back now yields initiative without ending the dispute."
    )

    def fake_transform_candidates(*args, **kwargs):
        source_row = kwargs["source_row"]
        if source_row["id"] == "sg_case_1":
            return [repeated_1]
        return [repeated_2, diverse_2]

    monkeypatch.setattr(module, "transform_source_row_candidates", fake_transform_candidates)

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--few-shot-path",
            str(fewshot),
            "--rubric-path",
            str(rubric),
            "--num-candidates",
            "2",
        ]
    )

    assert exit_code == 0
    success_rows = json.loads((output_dir / "escalation_game.success.json").read_text(encoding="utf-8"))
    candidates = [json.loads(line) for line in (output_dir / "escalation_game.candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    diversity_report = json.loads((output_dir / "diversity_report.json").read_text(encoding="utf-8"))

    selected_by_id = {row["provenance"]["id"]: row for row in success_rows}
    assert len(candidates) == 3
    assert selected_by_id["sg_case_2"]["description"] == diverse_2["description"]
    assert diversity_report["candidate_counts"] == {"generated": 3, "selected": 2}
    assert diversity_report["selected_description_metrics"]["distinct_2"] > 0


def test_run_transform_passes_request_timeout_to_candidate_generation(
    tmp_path: Path, prompt_assets, monkeypatch: pytest.MonkeyPatch
):
    # data_creation/transform_social_game_cases.py: forward CLI request timeout into candidate generation so hung model calls fail on the configured bound.
    from data_creation import transform_social_game_cases as module

    input_path = tmp_path / "escalation_game_cases.jsonl"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_jsonl(
        input_path,
        [
            _sample_source_row("sg_case_1", "game_1"),
            _sample_source_row("sg_case_2", "game_2"),
        ],
    )
    rubric, fewshot = prompt_assets
    seen_timeouts: list[float | None] = []

    def fake_transform_candidates(*args, **kwargs):
        seen_timeouts.append(kwargs.get("request_timeout_seconds"))
        source_row = kwargs["source_row"]
        return [_sample_escalation_transformed_row(source_row["id"], source_row["source"]["game_id"])]

    monkeypatch.setattr(module, "transform_source_row_candidates", fake_transform_candidates)

    exit_code = module.main(
        [
            "--social-game",
            "escalation_game",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--few-shot-path",
            str(fewshot),
            "--rubric-path",
            str(rubric),
            "--request-timeout-seconds",
            "17.5",
        ]
    )

    assert exit_code == 0
    assert seen_timeouts == [17.5, 17.5]


def test_compute_description_diversity_report_tracks_repeated_ngrams_and_distinct_scores():
    # data_creation/transform_social_game_cases.py: report classic n-gram diversity metrics for selected descriptions.
    from data_creation.transform_social_game_cases import compute_description_diversity_report

    report = compute_description_diversity_report(
        [
            "alpha beta gamma delta",
            "alpha beta gamma epsilon",
            "theta iota kappa lambda",
        ]
    )

    assert report["description_count"] == 3
    assert report["selected_description_metrics"]["distinct_1"] > 0
    assert report["selected_description_metrics"]["distinct_2"] > 0
    assert report["repeated_3grams"][0]["ngram"] == "alpha beta gamma"
    assert report["repeated_3grams"][0]["count"] == 2
