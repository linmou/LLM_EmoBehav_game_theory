"""
Enhanced adapter for LongBench long-context benchmark suite with comprehensive evaluation.
"""

import json
from typing import Any, Dict, List, Optional

# Import evaluation utilities
from .evaluation_utils import get_score_one

try:
    from .base_adapter import BenchmarkAdapter
    from .datasets import LongBenchDataset
    from ..data_models import BenchmarkConfig, BenchmarkItem
except ImportError:
    from emotion_memory_experiments.adapters.base_adapter import BenchmarkAdapter
    from emotion_memory_experiments.adapters.datasets import LongBenchDataset
    from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class LongBenchAdapter(BenchmarkAdapter):
    """Enhanced adapter for LongBench long-context benchmark suite with comprehensive evaluation"""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        
        # LongBench task-specific evaluation method mapping to official metrics
        self.task_evaluators = {
            "narrativeqa": "narrativeqa",
            "qasper": "qasper", 
            "multifieldqa_en": "multifieldqa_en",
            "multifieldqa_zh": "multifieldqa_zh",
            "hotpotqa": "hotpotqa",
            "2wikimqa": "2wikimqa",
            "musique": "musique",
            "dureader": "dureader",
            "gov_report": "gov_report",
            "qmsum": "qmsum",
            "multi_news": "multi_news",
            "vcsum": "vcsum",
            "trec": "trec",
            "triviaqa": "triviaqa",
            "samsum": "samsum",
            "lsht": "lsht",
            "passage_retrieval_en": "passage_retrieval_en",
            "passage_count": "passage_count",
            "passage_retrieval_zh": "passage_retrieval_zh",
            "lcc": "lcc",
            "repobench-p": "repobench-p",
        }
        
        # Task complexity classification
        self.simple_tasks = {
            "trec", "lsht", "passage_count", "passage_retrieval_en", 
            "passage_retrieval_zh"
        }
        self.complex_tasks = {
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
            "qmsum", "multi_news", "vcsum", "triviaqa", "samsum",
            "lcc", "repobench-p"
        }
        
        # Task metrics mapping
        self.task_metrics = {
            "narrativeqa": ["f1_score", "precision", "recall"],
            "qasper": ["f1_score", "precision", "recall"],
            "multifieldqa_en": ["f1_score", "precision", "recall"],
            "multifieldqa_zh": ["f1_score_zh"],
            "hotpotqa": ["f1_score", "precision", "recall"],
            "2wikimqa": ["f1_score", "precision", "recall"],
            "musique": ["f1_score", "precision", "recall"],
            "dureader": ["rouge_score"],
            "gov_report": ["rouge_score"],
            "qmsum": ["rouge_score"],
            "multi_news": ["rouge_score"],
            "vcsum": ["rouge_score"],
            "trec": ["accuracy"],
            "triviaqa": ["f1_score", "precision", "recall"],
            "samsum": ["rouge_score"],
            "lsht": ["accuracy"],
            "passage_retrieval_en": ["accuracy"],
            "passage_count": ["accuracy"],
            "passage_retrieval_zh": ["accuracy"],
            "lcc": ["code_similarity"],
            "repobench-p": ["code_similarity"],
        }

        # Tasks that need preprocessing (strip newlines, take first line)
        self.preprocessing_tasks = {"trec", "triviaqa", "samsum", "lsht"}

    def create_dataset(
        self, 
        prompt_wrapper=None,
        max_context_length: Optional[int] = None,
        tokenizer=None,
        truncation_strategy: str = "right"
    ) -> LongBenchDataset:
        """Create simple LongBench dataset with optional truncation"""
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")

        items: List[BenchmarkItem] = []

        if data_path.suffix == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        item_data = json.loads(line)
                        items.append(self._parse_longbench_item(item_data, i))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
                for i, item_data in enumerate(data_list):
                    items.append(self._parse_longbench_item(item_data, i))

        if self.config.sample_limit:
            items = items[: self.config.sample_limit]

        return LongBenchDataset(
            items, 
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy
        )

    def _parse_longbench_item(
        self, item_data: Dict[str, Any], index: int
    ) -> BenchmarkItem:
        """Parse a single LongBench item into standard format"""
        item_id = item_data.get("id", index)

        # LongBench typically has 'context' and 'input' fields
        context = item_data.get("context", "")
        input_text = item_data.get("input", item_data.get("question", ""))

        # Ground truth can be in different fields
        ground_truth = item_data.get(
            "answers", item_data.get("answer", item_data.get("target", None))
        )

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
        """Enhanced LongBench evaluation using official implementation"""
        # Apply preprocessing for specific tasks (from official LongBench eval.py)
        if task_name in self.preprocessing_tasks:
            response = response.lstrip('\n').split('\n')[0]
            
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
        from .evaluation_utils import longbench_qa_f1_score, f1_score, normalize_answer
        from collections import Counter
        
        # Apply preprocessing if needed
        if task_name in self.preprocessing_tasks:
            response = response.lstrip('\n').split('\n')[0]
            
        base_score = self.evaluate_response(response, ground_truth, task_name)
        
        # Return detailed metrics based on task type
        if task_name in ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]:
            # Calculate detailed F1 metrics for QA tasks
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
                    max_f1, max_precision, max_recall = float(f1), float(precision), float(recall)
            
            return {
                "f1_score": max_f1,
                "precision": max_precision,
                "recall": max_recall,
                "overall_score": base_score
            }
        elif task_name == "multifieldqa_zh":
            return {
                "f1_score_zh": base_score,
                "overall_score": base_score
            }
        elif task_name in ["dureader", "gov_report", "qmsum", "multi_news", "vcsum", "samsum"]:
            return {
                "rouge_score": base_score,
                "overall_score": base_score
            }
        elif task_name in ["trec", "lsht", "passage_retrieval_en", "passage_retrieval_zh", "passage_count"]:
            return {
                "accuracy": base_score,
                "overall_score": base_score
            }
        elif task_name in ["lcc", "repobench-p"]:
            return {
                "code_similarity": base_score,
                "overall_score": base_score
            }
        else:
            return {
                "score": base_score,
                "overall_score": base_score
            }

    def evaluate_by_length(
        self, 
        responses: List[str], 
        ground_truths: List[Any], 
        task_names: List[str],
        lengths: List[int]
    ) -> Dict[str, float]:
        """
        Enhanced LongBench-E style evaluation by context length categories.
        Implements exact logic from official LongBench eval.py
        """
        scores: Dict[str, List[float]] = {"0-4k": [], "4-8k": [], "8k+": []}
        
        for response, gt, task, length in zip(responses, ground_truths, task_names, lengths):
            # Apply preprocessing if needed
            if task in self.preprocessing_tasks:
                response = response.lstrip('\n').split('\n')[0]
                
            score = self.evaluate_response(response, gt, task)
            
            if length < 4000:
                scores["0-4k"].append(score)
            elif length < 8000:
                scores["4-8k"].append(score)
            else:
                scores["8k+"].append(score)
        
        # Compute average scores for each length category (exact LongBench logic)
        result = {}
        for category, score_list in scores.items():
            if score_list:
                result[category] = round(100 * sum(score_list) / len(score_list), 2)
            else:
                result[category] = 0.0
                
        return result