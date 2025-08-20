"""
Enhanced adapter for InfiniteBench tasks with comprehensive evaluation methods.
"""

import json
from typing import Any, Dict, List

# Import evaluation utilities
from .evaluation_utils import get_score_one

try:
    from .base_adapter import BenchmarkAdapter
    from .datasets import InfiniteBenchDataset
    from ..data_models import BenchmarkConfig, BenchmarkItem
except ImportError:
    from emotion_memory_experiments.adapters.base_adapter import BenchmarkAdapter
    from emotion_memory_experiments.adapters.datasets import InfiniteBenchDataset
    from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class InfiniteBenchAdapter(BenchmarkAdapter):
    """Enhanced adapter for InfiniteBench tasks with comprehensive evaluation methods"""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        
        # Complete task-specific evaluation method mapping from InfiniteBench
        self.task_evaluators = {
            "passkey": "passkey",
            "kv_retrieval": "kv_retrieval", 
            "kv_retrieval_prefix": "kv_retrieval",
            "kv_retrieval_both": "kv_retrieval",
            "number_string": "number_string",
            "longbook_qa_eng": "longbook_qa_eng",
            "longbook_sum_eng": "longbook_sum_eng",
            "longbook_choice_eng": "longbook_choice_eng",
            "longbook_qa_chn": "longbook_qa_chn",
            "longdialogue_qa_eng": "longdialogue_qa_eng",
            "code_debug": "code_debug",
            "code_run": "code_run",
            "math_calc": "math_calc",
            "math_find": "math_find",
        }

        # Evaluation complexity classification for pipeline optimization
        self.simple_tasks = {
            "passkey", "kv_retrieval", "kv_retrieval_prefix", "kv_retrieval_both",
            "number_string", "longbook_choice_eng", "math_calc", "math_find", "code_run"
        }
        self.complex_tasks = {
            "longbook_qa_eng", "longbook_sum_eng", "longbook_qa_chn", 
            "longdialogue_qa_eng", "code_debug"
        }
        
        # Task metrics mapping
        self.task_metrics = {
            "passkey": ["exact_match"],
            "kv_retrieval": ["exact_match"],
            "number_string": ["exact_match"],
            "longbook_qa_eng": ["f1_score", "precision", "recall"],
            "longbook_sum_eng": ["rouge_score"],
            "longbook_choice_eng": ["accuracy"],
            "longbook_qa_chn": ["f1_score_zh"],
            "longdialogue_qa_eng": ["exact_match"],
            "code_debug": ["pattern_match", "exact_match"],
            "code_run": ["numeric_match"],
            "math_calc": ["sequence_accuracy"],
            "math_find": ["numeric_match"],
        }

    def create_dataset(self, prompt_wrapper=None) -> InfiniteBenchDataset:
        """Create simple InfiniteBench dataset"""
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")

        items: List[BenchmarkItem] = []

        if data_path.suffix == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        item_data = json.loads(line)
                        items.append(self._parse_infinitebench_item(item_data, i))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
                for i, item_data in enumerate(data_list):
                    items.append(self._parse_infinitebench_item(item_data, i))

        if self.config.sample_limit:
            items = items[: self.config.sample_limit]

        return InfiniteBenchDataset(items, prompt_wrapper=prompt_wrapper)

    def _parse_infinitebench_item(
        self, item_data: Dict[str, Any], index: int
    ) -> BenchmarkItem:
        """Parse a single InfiniteBench item into standard format"""
        # InfiniteBench format varies by task, try common fields
        item_id = item_data.get("id", index)

        # Handle different input formats
        input_text = item_data.get("input", "")
        context = item_data.get("context", "")

        # Some tasks combine context and input
        if not input_text and context:
            input_text = context
            context = None

        # Ground truth can be in different fields
        ground_truth = item_data.get(
            "answer", item_data.get("label", item_data.get("ground_truth", None))
        )

        # Handle list format (like ["71432"] -> "71432")
        if isinstance(ground_truth, list) and len(ground_truth) == 1:
            ground_truth = ground_truth[0]

        return BenchmarkItem(
            id=item_id,
            input_text=input_text,
            context=context,
            ground_truth=ground_truth,
            metadata=item_data,
        )

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str
    ) -> float:
        """Enhanced evaluation using official InfiniteBench implementation"""
        # Use unified evaluation function from evaluation_utils
        return get_score_one(response, ground_truth, task_name, "emotion_model")

    def get_evaluation_complexity(self, task_name: str) -> str:
        """Determine evaluation complexity for pipeline optimization"""
        if task_name in self.simple_tasks:
            return "simple"
        elif task_name in self.complex_tasks:
            return "complex"
        return "simple"

    def get_task_metrics(self, task_name: str) -> List[str]:
        """Return list of metrics available for this task"""
        return self.task_metrics.get(task_name, ["accuracy"])

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> Dict[str, float]:
        """Return detailed metrics for comprehensive analysis"""
        from .evaluation_utils import qa_f1_score, qa_f1_score_zh, f1_score, normalize_answer, normalize_zh_answer
        from collections import Counter
        
        base_score = self.evaluate_response(response, ground_truth, task_name)
        
        # Return detailed metrics based on task type
        if task_name in ["longbook_qa_eng", "longbook_qa_chn"]:
            # Calculate detailed F1 metrics
            if task_name == "longbook_qa_eng":
                if not isinstance(ground_truth, list):
                    ground_truth = [ground_truth]
                
                max_f1, max_precision, max_recall = 0, 0, 0
                for gt in ground_truth:
                    normalized_prediction = normalize_answer(response)
                    normalized_ground_truth = normalize_answer(str(gt))
                    prediction_tokens = normalized_prediction.split()
                    ground_truth_tokens = normalized_ground_truth.split()
                    f1, precision, recall = f1_score(prediction_tokens, ground_truth_tokens)
                    if f1 > max_f1:
                        max_f1, max_precision, max_recall = f1, precision, recall
                
                return {
                    "f1_score": max_f1,
                    "precision": max_precision,
                    "recall": max_recall,
                    "overall_score": base_score
                }
            else:  # Chinese QA
                return {
                    "f1_score_zh": base_score,
                    "overall_score": base_score
                }
        elif task_name == "longbook_sum_eng":
            return {
                "rouge_score": base_score,
                "overall_score": base_score
            }
        elif task_name == "math_calc":
            return {
                "sequence_accuracy": base_score,
                "overall_score": base_score
            }
        else:
            return {
                "accuracy": base_score,
                "overall_score": base_score
            }