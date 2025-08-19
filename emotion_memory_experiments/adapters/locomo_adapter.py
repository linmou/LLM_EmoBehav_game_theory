"""
Adapter for LoCoMo conversational memory benchmark.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .base_adapter import BenchmarkAdapter
    from .datasets import LoCoMoDataset
    from ..data_models import BenchmarkConfig, BenchmarkItem
except ImportError:
    from emotion_memory_experiments.adapters.base_adapter import BenchmarkAdapter
    from emotion_memory_experiments.adapters.datasets import LoCoMoDataset
    from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class LoCoMoAdapter(BenchmarkAdapter):
    """Adapter for LoCoMo conversational memory benchmark"""

    def create_dataset(self, prompt_wrapper=None) -> LoCoMoDataset:
        """Load LoCoMo data as PyTorch Dataset"""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items: List[BenchmarkItem] = []
        for sample in data:
            sample_id = sample.get("sample_id", len(items))
            conversation = sample.get("conversation", {})
            qa_pairs = sample.get("qa", [])

            for qa in qa_pairs:
                items.append(
                    BenchmarkItem(
                        id=f"{sample_id}_{len(items)}",
                        input_text=qa["question"],
                        context=self._format_conversation(conversation),
                        ground_truth=qa["answer"],
                        metadata={
                            "sample_id": sample_id,
                            "category": qa.get("category", "unknown"),
                            "evidence": qa.get("evidence", []),
                        },
                    )
                )

        if self.config.sample_limit:
            items = items[: self.config.sample_limit]

        return LoCoMoDataset(items, prompt_wrapper=prompt_wrapper)

    def _format_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format LoCoMo conversation data into context string"""
        formatted_sessions = []

        # Extract sessions in chronological order
        session_keys = [
            k
            for k in conversation.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        ]
        session_keys.sort()

        for session_key in session_keys:
            session_data = conversation[session_key]
            date_key = f"{session_key}_date_time"
            session_date = conversation.get(date_key, "Unknown date")

            formatted_sessions.append(f"=== {session_key.upper()} ({session_date}) ===")

            for turn in session_data:
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                formatted_sessions.append(f"{speaker}: {text}")

            formatted_sessions.append("")  # Empty line between sessions

        return "\n".join(formatted_sessions)

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str
    ) -> float:
        """ORIGINAL LoCoMo evaluation: F1 score with stemming"""
        return self._locomo_f1_score(response, ground_truth)

    def _locomo_f1_score(self, prediction: str, ground_truth: str) -> float:
        """Original LoCoMo F1 score implementation with stemming"""
        import re
        import string
        from collections import Counter

        def normalize_answer(s):
            """Normalize answer following LoCoMo method"""
            s = str(s).replace(",", "")

            def remove_articles(text):
                return re.sub(r"\b(a|an|the|and)\b", " ", text)

            def white_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        # Simple stemming (basic version without NLTK dependency)
        def simple_stem(word):
            # Basic English stemming rules
            if word.endswith("ing"):
                return word[:-3]
            elif word.endswith("ed"):
                return word[:-2]
            elif word.endswith("s") and len(word) > 2:
                return word[:-1]
            return word

        # Normalize and stem tokens
        prediction_tokens = [
            simple_stem(w) for w in normalize_answer(prediction).split()
        ]
        ground_truth_tokens = [
            simple_stem(w) for w in normalize_answer(ground_truth).split()
        ]

        # Compute F1 score
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = (
            1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0.0
        )
        recall = (
            1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
        )

        if precision + recall == 0:
            return 0.0

        f1 = (2 * precision * recall) / (precision + recall)
        return f1