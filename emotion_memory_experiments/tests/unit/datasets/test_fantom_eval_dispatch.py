"""
Tests for FantomDataset evaluation dispatch and behaviors.

Covers:
- Binary: synonyms and wrong mappings
- Choice: letter mapping with parentheses
- List: JSON list and missing element fail
- Text (fact/gen): uses token F1 (expects partial credit for overlap)

Responsible files:
- emotion_memory_experiments/datasets/fantom.py
"""

import unittest

from emotion_memory_experiments.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_memory_experiments.data_models import BenchmarkConfig


def _cfg(task: str) -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        name="fantom",
        task_type=task,
        data_path=None,
        base_data_dir="data/fantom",
        sample_limit=3,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )
    cfg.data_path = cfg.get_data_path()
    return cfg


class TestFantomEvalDispatch(unittest.TestCase):
    def _build(self, task: str):
        class _PF:
            def build(self, system_prompt: str, user_messages, **kwargs):
                return f"{system_prompt}\n{user_messages}"

        cfg = _cfg(task)
        return create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=_PF(),
        )

    def test_binary_synonyms_and_fail(self):
        _, _, ds = self._build("short_infoaccessibility_binary_inaccessible")
        ex = ds[0]
        gt = ex["ground_truth"]
        # map synonyms: true/false, knows/does not know
        if gt == "yes":
            self.assertEqual(
                ds.evaluate_response("True", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                1.0,
            )
            self.assertEqual(
                ds.evaluate_response("no", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                0.0,
            )
        else:
            self.assertEqual(
                ds.evaluate_response("does not know", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                1.0,
            )
            self.assertEqual(
                ds.evaluate_response("yes", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                0.0,
            )

    def test_choice_letter_mapping_pass_and_fail(self):
        _, _, ds = self._build("short_belief_choice_inaccessible")
        ex = ds[0]
        md = ex["item"].metadata  # type: ignore[attr-defined]
        correct_idx = md.get("correct_index")
        # Build letter answers
        correct_letter = chr(ord("a") + int(correct_idx))
        wrong_letter = chr(ord("a") + ((int(correct_idx) + 1) % len(md.get("options", []))))
        self.assertEqual(
            ds.evaluate_response(f"({correct_letter})", ex["ground_truth"], "short_belief_choice_inaccessible", ex["prompt"]),
            1.0,
        )
        self.assertEqual(
            ds.evaluate_response(f"{wrong_letter}.", ex["ground_truth"], "short_belief_choice_inaccessible", ex["prompt"]),
            0.0,
        )

    def test_list_json_and_missing_element(self):
        _, _, ds = self._build("short_answerability_list_inaccessible")
        ex = ds[0]
        gt = ex["ground_truth"]
        # JSON form should pass
        resp = {"answer": gt}
        self.assertEqual(
            ds.evaluate_response(str(resp), gt, "short_answerability_list_inaccessible", ex["prompt"]),
            1.0,
        )
        # Missing element should fail (not equal set)
        if len(gt) > 1:
            bad = gt[:-1]
        else:
            bad = ["nonexistent"]
        self.assertEqual(
            ds.evaluate_response(
                ", ".join(bad), gt, "short_answerability_list_inaccessible", ex["prompt"]
            ),
            0.0,
        )

        # If response includes a wrong item, it should fail even if all correct are present
        wrong = ex["item"].metadata.get("wrong_answer")  # type: ignore[attr-defined]
        if wrong and isinstance(wrong, list) and wrong:
            combined = gt + [wrong[0]]
            resp2 = {"answer": combined}
            self.assertEqual(
                ds.evaluate_response(str(resp2), gt, "short_answerability_list_inaccessible", ex["prompt"]),
                0.0,
            )

    def test_fact_and_gen_eval(self):
        # fact
        _, _, ds_fact = self._build("short_fact")
        exf = ds_fact[0]
        gt_f = exf["ground_truth"]
        self.assertEqual(
            ds_fact.evaluate_response(gt_f, gt_f, "short_fact", exf["prompt"]), 1.0
        )
        self.assertLess(
            ds_fact.evaluate_response(gt_f + " extra", gt_f, "short_fact", exf["prompt"]),
            1.0,
        )

        # gen: should prefer similarity to correct_answer over wrong_answer
        _, _, ds_gen = self._build("short_belief_gen_inaccessible")
        exg = ds_gen[0]
        gt_g = exg["ground_truth"]
        wrong = exg["item"].metadata.get("wrong_answer")  # type: ignore[attr-defined]
        # Force a deterministic lightweight embedder by monkeypatching _get_embedder
        class _Dummy:
            def encode(self, s):
                # Very simple bag length as 1-dim vector
                return __import__('numpy').array([len(str(s).split())], dtype=float)
        ds_gen._embedder = _Dummy()  # type: ignore[attr-defined]

        # Exact correct should be 1.0
        self.assertEqual(
            ds_gen.evaluate_response(gt_g, gt_g, "short_belief_gen_inaccessible", exg["prompt"]),
            1.0,
        )
        if wrong:
            # A response identical to wrong_answer should be scored 0.0
            self.assertEqual(
                ds_gen.evaluate_response(wrong, gt_g, "short_belief_gen_inaccessible", exg["prompt"]),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
