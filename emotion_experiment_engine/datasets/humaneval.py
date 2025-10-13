"""
HumanEvalDataset - thin adapter around upstream HumanEval problems and checker.

Loads /home/jjl7137/human-eval/data/HumanEval.jsonl.gz (or configured path),
exposes items as BenchmarkItem, and evaluates completions via
human_eval.execution.check_correctness with a short timeout.
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


def _strip_code_fences(completion: str) -> str:
    """Remove leading/trailing ``` fences (optionally with language tags)."""
    if not completion:
        return completion

    trimmed = completion.strip()
    if not trimmed.startswith("```"):
        return completion

    body = trimmed[3:]
    newline_idx = body.find("\n")
    if newline_idx == -1:
        return ""
    # Drop optional language identifier (e.g. python)
    body = body[newline_idx + 1 :]

    body = body.rstrip()
    if body.endswith("```"):
        body = body[:-3]
    return body.strip("\r\n")


def _import_humaneval() -> Any:
    """Import upstream human_eval module, adding local path if needed.

    We try normal import; if it fails, add the known checkout path
    '/home/jjl7137/human-eval' to sys.path and retry.
    """
    try:
        import human_eval  # type: ignore
        return human_eval
    except Exception:
        he_path = "/home/jjl7137/human-eval"
        if he_path not in sys.path:
            sys.path.insert(0, he_path)
        import human_eval  # type: ignore
        return human_eval


class HumanEvalDataset(BaseBenchmarkDataset):
    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper: Optional[Callable],
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper: Optional[Callable] = None,
        eval_timeout: float = 3.0,
    ):
        self.eval_timeout = float(eval_timeout)
        super().__init__(
            config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"HumanEval data not found: {data_path}")

        items: List[BenchmarkItem] = []
        # HumanEval canonical file is gz jsonl
        if str(data_path).endswith(".gz"):
            with gzip.open(str(data_path), "rt") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    items.append(
                        BenchmarkItem(
                            id=row["task_id"],
                            input_text=row["prompt"],
                            context=None,
                            ground_truth=row,
                            metadata={
                                "entry_point": row.get("entry_point"),
                                "source": "humaneval",
                            },
                        )
                    )
        else:
            # Fallback: treat as plain jsonl
            with open(data_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    items.append(
                        BenchmarkItem(
                            id=row["task_id"],
                            input_text=row["prompt"],
                            context=None,
                            ground_truth=row,
                            metadata={
                                "entry_point": row.get("entry_point"),
                                "source": "humaneval",
                            },
                        )
                    )

        return items

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        he = _import_humaneval()
        from human_eval.execution import check_correctness  # type: ignore

        cleaned_response = _strip_code_fences(response)

        result = check_correctness(
            problem=ground_truth, completion=cleaned_response, timeout=self.eval_timeout, completion_id=None
        )
        return 1.0 if bool(result.get("passed")) else 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]

    def compute_split_metrics(self, records: List["ResultRecord"]) -> Dict[str, float]:
        # Basic pass rate (pass@1)
        total = len(records)
        if total == 0:
            return {"pass_rate": 0.0}
        ok = 0
        for r in records:
            s = r.score
            try:
                ok += 1 if s and float(s) >= 1.0 else 0
            except Exception:
                pass
        return {"pass_rate": ok / float(total)}
