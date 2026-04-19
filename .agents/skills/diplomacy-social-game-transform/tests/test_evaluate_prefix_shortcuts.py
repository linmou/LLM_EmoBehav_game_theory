#!/usr/bin/env python3
# Purpose: test .agents/skills/diplomacy-social-game-transform/scripts/evaluate_prefix_shortcuts.py for generic two-choice label extraction and legacy label compatibility.

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_prefix_shortcuts.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_prefix_shortcuts",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluatePrefixShortcutsTests(unittest.TestCase):
    def test_iter_labeled_choices_supports_cooperate_defect_few_shot_rows(self):
        # .agents/skills/diplomacy-social-game-transform/scripts/evaluate_prefix_shortcuts.py: accept Prisoners' Dilemma-style behavior choices from few-shot outputs.
        module = _load_module()
        payload = [
            {
                "input": {"variant_name": "TWO_AGENTS_SINGLE_TURN"},
                "output": {
                    "behavior_choices": {
                        "cooperate": "Hold the Belgium corridor outside your chief advance this season.",
                        "defect": "Hold the Belgium corridor inside your chief advance this season.",
                    }
                },
            }
        ]

        self.assertEqual(
            module.iter_labeled_choices(payload),
            [
                (
                    "hold the loc corridor outside your chief advance this season",
                    "cooperate",
                ),
                (
                    "hold the loc corridor inside your chief advance this season",
                    "defect",
                ),
            ],
        )

    def test_iter_labeled_choices_keeps_legacy_escalate_withdraw_rows(self):
        # .agents/skills/diplomacy-social-game-transform/scripts/evaluate_prefix_shortcuts.py: keep support for escalation-game style transformed corpora.
        module = _load_module()
        payload = [
            {
                "behavior_choices": {
                    "escalate": "Push harder into the border district this season.",
                    "withdraw": "Keep your pressure away from the border district this season.",
                }
            }
        ]

        self.assertEqual(
            module.iter_labeled_choices(payload),
            [
                (
                    "push harder into the border district this season",
                    "escalate",
                ),
                (
                    "keep your pressure away from the border district this season",
                    "withdraw",
                ),
            ],
        )

    def test_main_reports_metrics_for_cooperate_defect_file(self):
        # .agents/skills/diplomacy-social-game-transform/scripts/evaluate_prefix_shortcuts.py: score generic two-choice files through the CLI path.
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "prisoners_dilemma_few_shot_examples.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "output": {
                                "behavior_choices": {
                                    "cooperate": "Leave the corridor outside your main effort this season.",
                                    "defect": "Leave the corridor inside your main effort this season.",
                                }
                            }
                        },
                        {
                            "output": {
                                "behavior_choices": {
                                    "cooperate": "Keep the route beyond your active push this season.",
                                    "defect": "Keep the route within your active push this season.",
                                }
                            }
                        },
                    ]
                ),
                encoding="utf-8",
            )

            original_parse_args = module.parse_args
            module.parse_args = lambda: type("Args", (), {"json_path": str(json_path)})()
            stdout = io.StringIO()
            try:
                with redirect_stdout(stdout):
                    self.assertEqual(module.main(), 0)
            finally:
                module.parse_args = original_parse_args

        output = stdout.getvalue()
        self.assertIn("samples=4", output)
        self.assertIn("first_token_accuracy=", output)
        self.assertIn("first_two_token_accuracy=", output)


if __name__ == "__main__":
    unittest.main()
