"""
FantomDataset - Minimal integration for FANToM benchmark tasks.

Implements two easiest tasks with straightforward evaluation:
- short_answerability_binary_inaccessible: binary yes/no
- short_belief_choice_inaccessible: 2-option multiple choice

JSONL expected formats:

1) short_answerability_binary_inaccessible
   {"context": str, "question": str, "answer": "yes"|"no"}

2) short_belief_choice_inaccessible
   {"context": str, "question": str, "options": [str, str, ...], "correct_index": int}

Evaluation rules are simple, case-insensitive, and accept either option text
or A/B/C... letter choices produced by MemoryPromptWrapper.
"""

import json
from typing import Any, Dict, List, Optional

from .base import BaseBenchmarkDataset
from ..data_models import BenchmarkItem


class FantomDataset(BaseBenchmarkDataset):
    """Minimal dataset for FANToM with two easy tasks."""

    SUPPORTED_TASKS = {
        "short_answerability_binary_inaccessible": "_load_answerability_binary",
        "short_belief_choice_inaccessible": "_load_belief_choice",
    }

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        task = self.config.task_type
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported FANToM task: {task}. Supported: {list(self.SUPPORTED_TASKS.keys())}"
            )

        # Load raw data (JSON or JSONL) using base loader, then parse per task
        raw = self._load_raw_data()
        loader = getattr(self, self.SUPPORTED_TASKS[task])
        return loader(raw)

    def _load_answerability_binary(self, raw: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                continue
            context = ex.get("context", "")
            question = ex.get("question", "")
            ans = ex.get("answer")
            if not isinstance(context, str) or not isinstance(question, str) or not isinstance(ans, str):
                continue
            ans_norm = ans.strip().lower()
            if ans_norm not in {"yes", "no"}:
                # Skip invalid entries
                continue

            items.append(
                BenchmarkItem(
                    id=f"fantom_ab_{i}",
                    context=context,
                    input_text=question,
                    ground_truth=ans_norm,
                    metadata={
                        "task": "short_answerability_binary_inaccessible",
                        "options": ["Yes", "No"],
                    },
                )
            )
        if not items:
            raise ValueError("No valid items loaded for answerability_binary task")
        return items

    def _load_belief_choice(self, raw: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                continue
            context = ex.get("context", "")
            question = ex.get("question", "")
            options = ex.get("options")
            correct_index = ex.get("correct_index")
            if (
                not isinstance(context, str)
                or not isinstance(question, str)
                or not isinstance(options, list)
                or len(options) < 2
                or not isinstance(correct_index, int)
                or not (0 <= correct_index < len(options))
            ):
                continue
            # Normalize option texts
            norm_options = [str(o).strip() for o in options]
            correct_text = norm_options[correct_index]

            items.append(
                BenchmarkItem(
                    id=f"fantom_bc_{i}",
                    context=context,
                    input_text=question,
                    # Store canonical correct as list to align with MC evaluation style
                    ground_truth=[correct_text],
                    metadata={
                        "task": "short_belief_choice_inaccessible",
                        "options": norm_options,
                        "correct_index": correct_index,
                    },
                )
            )
        if not items:
            raise ValueError("No valid items loaded for belief_choice task")
        return items

    def _parse_letter_choice(self, response: str, options: List[str]) -> Optional[str]:
        """Map letter choices A/B/C... to option text, return None if no mapping."""
        if not response:
            return None
        r = response.strip().upper()
        if not r:
            return None
        first = r[0]
        if "A" <= first <= "Z":
            idx = ord(first) - ord("A")
            if 0 <= idx < len(options):
                return options[idx]
        return None

    def _extract_answer_field(self, response: str) -> Optional[str]:
        """Try to extract the 'answer' field from a JSON-like response.

        Supports standard JSON (double quotes) and Python-like dict strings
        (single quotes). Falls back to regex if JSON parsing fails.
        """
        if not response or not isinstance(response, str):
            return None
        txt = response.strip()

        # Fast path: if clearly not an object, return None
        if not (txt.startswith("{") and txt.endswith("}")):
            return None

        # Try JSON first
        try:
            import json as _json

            data = _json.loads(txt)
            if isinstance(data, dict):
                for key in ("answer", "predict answer", "predicted_answer", "prediction"):
                    if key in data:
                        ans = data[key]
                        return str(ans) if ans is not None else None
        except Exception:
            pass

        # Try Python literal dict parsing
        try:
            import ast as _ast

            data = _ast.literal_eval(txt)
            if isinstance(data, dict):
                for key in ("answer", "predict answer", "predicted_answer", "prediction"):
                    if key in data:
                        ans = data[key]
                        return str(ans) if ans is not None else None
        except Exception:
            pass

        # Regex fallback for potential keys and values
        try:
            import re as _re
            for key in ("answer", "predict answer", "predicted_answer", "prediction"):
                m = _re.search(
                    rf"['\"]{_re.escape(key)}['\"]\s*:\s*['\"]([^'\"]+)['\"]",
                    txt,
                )
                if m:
                    return m.group(1)
        except Exception:
            pass

        return None

    def _extract_index_fields(self, response: str) -> List[int]:
        """Extract numeric indices from JSON-like fields such as 'answer_index' or 'option_index'."""
        indices: List[int] = []
        if not response or not isinstance(response, str):
            return indices
        txt = response.strip()
        if not (txt.startswith("{") and txt.endswith("}")):
            return indices
        # Try JSON then literal eval
        for parser in ("json", "ast"):
            try:
                if parser == "json":
                    import json as _json
                    data = _json.loads(txt)
                else:
                    import ast as _ast
                    data = _ast.literal_eval(txt)
                if isinstance(data, dict):
                    for key in ("answer_index", "option_index", "indices", "index"):
                        val = data.get(key)
                        if isinstance(val, list):
                            for v in val:
                                try:
                                    indices.append(int(v))
                                except Exception:
                                    pass
                        elif isinstance(val, (int, float)):
                            indices.append(int(val))
                break
            except Exception:
                continue
        return indices

    def _parse_options_from_prompt(self, prompt: str) -> List[str]:
        options: List[str] = []
        if "Options:" not in prompt:
            return options
        try:
            import re as _re
            tail = prompt.split("Options:", 1)[1]
            tail = tail.replace("\\n", "\n").strip()
            if tail.startswith("['"):
                tail = tail[2:]
            if tail.endswith("']"):
                tail = tail[:-2]
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            for ln in lines:
                m = _re.match(r"^[^A-Za-z0-9]*([A-Z]|\d+)\.?\s+(.*)$", ln)
                if m:
                    options.append(m.group(2).strip())
        except Exception:
            return []
        return options

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        task = task_name
        # Extract 'answer' if response is in JSON format
        extracted = self._extract_answer_field(response)
        response_norm = (extracted if extracted is not None else (response or "")).strip()

        if task == "short_answerability_binary_inaccessible":
            if not isinstance(ground_truth, str):
                raise TypeError("ground_truth must be a string for binary task")
            gt = ground_truth.strip().lower()
            resp = response_norm.strip()
            resp_lower = resp.lower()

            # Accept explicit yes/no anywhere in response
            if "yes" in resp_lower:
                pred = "yes"
            elif "no" in resp_lower:
                pred = "no"
            else:
                # Try mapping via options parsed from prompt
                options = self._parse_options_from_prompt(prompt)
                mapped_pred = None
                if options:
                    # Normalize option values
                    norm_opts = [o.strip().lower() for o in options]
                    # Letter form: A/B with optional dot and tail
                    import re as _re
                    m_letter = _re.match(r"^\s*([A-Za-z])\.?\s*(.*)$", resp)
                    if m_letter:
                        idx = ord(m_letter.group(1).upper()) - ord("A")
                        tail = m_letter.group(2).strip().lower()
                        if 0 <= idx < len(norm_opts):
                            opt_text = norm_opts[idx]
                            if "yes" in opt_text or opt_text == "yes":
                                mapped_pred = "yes"
                            elif "no" in opt_text or opt_text == "no":
                                mapped_pred = "no"
                            # If tail includes yes/no directly
                            elif tail == "yes" or tail == "no":
                                mapped_pred = tail

                    # Numeric form: 1/2 (1-based)
                    if mapped_pred is None:
                        m_num = _re.match(r"^\s*(\d+)\.?\s*(.*)$", resp)
                        if m_num:
                            idx = int(m_num.group(1)) - 1
                            tail = m_num.group(2).strip().lower()
                            if 0 <= idx < len(norm_opts):
                                opt_text = norm_opts[idx]
                                if "yes" in opt_text or opt_text == "yes":
                                    mapped_pred = "yes"
                                elif "no" in opt_text or opt_text == "no":
                                    mapped_pred = "no"
                                elif tail == "yes" or tail == "no":
                                    mapped_pred = tail

                # If still not mapped, use conventional fallback A/1→yes, B/2→no
                if mapped_pred is None:
                    if resp_lower.startswith("a") or resp_lower.startswith("1"):
                        mapped_pred = "yes"
                    elif resp_lower.startswith("b") or resp_lower.startswith("2"):
                        mapped_pred = "no"

                pred = mapped_pred if mapped_pred is not None else resp_lower
            return 1.0 if pred == gt else 0.0

        if task == "short_belief_choice_inaccessible":
            if not isinstance(ground_truth, list) or not ground_truth:
                raise TypeError("ground_truth must be non-empty list for choice task")
            correct_text = ground_truth[0].strip().lower()
            resp = response_norm.strip()
            resp_lower = resp.lower()

            # 1) Exact option content
            if resp_lower == correct_text:
                return 1.0
            # 2) Content with prefix label (e.g., "B. <text>")
            if correct_text and correct_text in resp_lower:
                return 1.0

            # 3) Map letters/numbers to options from prompt
            options = self._parse_options_from_prompt(prompt)
            if options:
                norm_options = [o.strip() for o in options]
                import re as _re
                # Letter: A or A. tail
                m_letter = _re.match(r"^\s*([A-Za-z])\.?\s*(.*)$", resp)
                if m_letter:
                    idx = ord(m_letter.group(1).upper()) - ord("A")
                    tail = m_letter.group(2).strip()
                    if 0 <= idx < len(norm_options):
                        mapped = norm_options[idx]
                        if mapped.strip().lower() == correct_text:
                            return 1.0
                        if tail and tail.strip().lower() == correct_text:
                            return 1.0
                # Numeric: 1 or 1. tail (assume 1-based)
                m_num = _re.match(r"^\s*(\d+)\.?\s*(.*)$", resp)
                if m_num:
                    idx = int(m_num.group(1)) - 1
                    tail = m_num.group(2).strip()
                    if 0 <= idx < len(norm_options):
                        mapped = norm_options[idx]
                        if mapped.strip().lower() == correct_text:
                            return 1.0
                        if tail and tail.strip().lower() == correct_text:
                            return 1.0

                # 4) JSON index fields
                indices = self._extract_index_fields(response)
                for n in indices:
                    idx = n - 1  # treat as 1-based
                    if 0 <= idx < len(norm_options):
                        mapped = norm_options[idx]
                        if mapped.strip().lower() == correct_text:
                            return 1.0

            return 0.0

        raise ValueError(
            f"Unsupported task for evaluation: {task}. Supported: {list(self.SUPPORTED_TASKS.keys())}"
        )

    def get_task_metrics(self, task_name: str) -> List[str]:
        if task_name not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task type: {task_name}. Supported: {list(self.SUPPORTED_TASKS.keys())}"
            )
        return ["accuracy"]
