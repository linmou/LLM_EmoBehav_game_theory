"""Tests for scripts/sc2_intentmap_to_sealed_auction.sh and Python defaults/prompt.
Responsible file: auto_experiments/sc2_dataset/generate_sc2_scenarios_from_intent_map_with_gemini.py
Purpose: Ensure CLI defaults and overwrite prompt behavior are correct."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_experiments.sc2_dataset import generate_sc2_scenarios_from_intent_map_with_gemini as mod


def test_shell_wrapper_exists_and_calls_python_module() -> None:
    script = Path("scripts/sc2_intentmap_to_sealed_auction.sh")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "python -m auto_experiments.sc2_dataset.generate_sc2_scenarios_from_intent_map_with_gemini" in text
    assert "--game_name Sealed_Auction" in text
    assert "--intent_map_jsonl datasets/intent_map_dataset_air72_base96p6_drop94p6_gold50.jsonl" in text
    assert (
        "--out data_creation/scenario_creation/langgraph_creation/SC2_Sealed_Auction_all_data_samples.json"
        in text
    )
    assert "--concurrency 20" in text


def test_argparse_defaults_match_requested_values() -> None:
    p = mod.build_arg_parser()
    args = p.parse_args([])
    assert args.intent_map_jsonl.as_posix() == "datasets/intent_map_dataset_air72_base96p6_drop94p6_gold50.jsonl"
    assert args.out.as_posix() == "data_creation/scenario_creation/langgraph_creation/SC2_Sealed_Auction_all_data_samples.json"
    assert args.concurrency == 20
    assert args.game_name == "Sealed_Auction"


def test_confirm_overwrite_prompts_when_file_exists(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    out.write_text("x", encoding="utf-8")
    assert mod._confirm_overwrite(out, input_fn=lambda _: "n") is False
    assert mod._confirm_overwrite(out, input_fn=lambda _: "y") is True


def test_confirm_overwrite_no_prompt_when_missing(tmp_path: Path) -> None:
    out = tmp_path / "missing.json"
    assert mod._confirm_overwrite(out, input_fn=lambda _: pytest.fail("should not prompt")) is True
