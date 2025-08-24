"""
LongBenchDataset - Specialized dataset for LongBench evaluation.
Handles LongBench-specific data format and routes to appropriate metric evaluators.
"""

from typing import Any, Dict, List
from ..data_models import BenchmarkConfig, BenchmarkItem
from ..evaluation_utils import get_score_one
from .base import BaseBenchmarkDataset

# Import all LongBench evaluators
from .. import evaluation_utils


class LongBenchDataset(BaseBenchmarkDataset):
    """
    Specialized dataset for LongBench tasks.
    
    Routes evaluation to metric-specific evaluators:
    - narrativeqa/qasper: F1 score with token normalization
    - gov_report/qmsum: ROUGE-L scoring
    - trec/lsht: Classification accuracy with substring matching
    - passage_count: Count accuracy with number extraction
    - lcc/repobench: Code similarity scoring
    - Chinese tasks: Character-level evaluation
    """
    
    # Static mapping eliminates if-else chains
    # Maps task names to evaluator function names in evaluation_utils
    METRIC_EVALUATORS = {
        "narrativeqa": "longbench_qa_f1_score",
        "qasper": "longbench_qa_f1_score",
        "multifieldqa_en": "longbench_qa_f1_score",
        "multifieldqa_zh": "longbench_qa_f1_zh_score", 
        "hotpotqa": "longbench_qa_f1_score",
        "2wikimqa": "longbench_qa_f1_score",
        "musique": "longbench_qa_f1_score",
        "dureader": "rouge_zh_score",
        "gov_report": "rouge_score",
        "qmsum": "rouge_score",
        "multi_news": "rouge_score",
        "vcsum": "rouge_zh_score",
        "trec": "classification_score",
        "triviaqa": "longbench_qa_f1_score", 
        "samsum": "rouge_score",
        "lsht": "classification_score",
        "passage_retrieval_en": "retrieval_score",
        "passage_count": "count_score",
        "passage_retrieval_zh": "retrieval_zh_score",
        "lcc": "code_sim_score",
        "repobench-p": "code_sim_score"
    }
    
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """Load LongBench-specific data format"""
        raw_data = self._load_raw_data()
        return self._parse_longbench_data(raw_data)
    
    def _parse_longbench_data(self, raw_data: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        """Parse LongBench format to BenchmarkItem objects"""
        items = []
        for i, item_data in enumerate(raw_data):
            item_id = item_data.get("id", f"longbench_{i}")
            input_text = item_data.get("input", "")
            context = item_data.get("context", "")
            
            # LongBench may have answers as list - take first one as primary
            ground_truth = item_data.get("answers", item_data.get("answer", None))
            if isinstance(ground_truth, list) and ground_truth:
                ground_truth = ground_truth[0]
            
            items.append(BenchmarkItem(
                id=item_id,
                input_text=input_text,
                context=context,
                ground_truth=ground_truth,
                metadata=item_data
            ))
        
        return items
    
    def evaluate_response(self, response: str, ground_truth: Any, task_name: str) -> float:
        """Route to metric-specific LongBench evaluator"""
        return self._route_to_evaluator(task_name, response, ground_truth)
    
    def _route_to_evaluator(self, task_name: str, response: str, ground_truth: Any) -> float:
        """Route to specific evaluator without if-else chains"""
        evaluator_name = self.METRIC_EVALUATORS.get(task_name)
        
        if not evaluator_name:
            # Fallback to generic evaluator for unknown tasks
            return float(get_score_one(response, ground_truth, task_name, "emotion_model"))
        
        # Get the evaluator function from evaluation_utils
        evaluator_func = getattr(evaluation_utils, evaluator_name)
        
        # Call the evaluator - these return floats directly
        result = evaluator_func(response, ground_truth)
        return float(result)
    
    def get_task_metrics(self, task_name: str) -> List[str]:
        """Return metrics available for LongBench task"""
        
        # QA F1 tasks
        if task_name in ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
                         "hotpotqa", "2wikimqa", "musique", "triviaqa"]:
            return ["f1_score", "precision", "recall"]
        
        # ROUGE scoring tasks  
        elif task_name in ["gov_report", "qmsum", "multi_news", "samsum", "dureader", "vcsum"]:
            return ["rouge_score"]
        
        # Classification tasks
        elif task_name in ["trec", "lsht"]:
            return ["classification_accuracy"]
        
        # Retrieval tasks
        elif task_name in ["passage_retrieval_en", "passage_retrieval_zh"]:
            return ["retrieval_accuracy"]
        
        # Count tasks
        elif task_name == "passage_count":
            return ["count_accuracy"]
        
        # Code tasks
        elif task_name in ["lcc", "repobench-p"]:
            return ["code_similarity"]
        
        # Default
        else:
            return ["accuracy"]
    
    def evaluate_by_length(self, responses: List[str], ground_truths: List[Any], 
                          context_lengths: List[int], task_names: List[str]) -> Dict[str, float]:
        """
        Evaluate responses grouped by context length ranges.
        LongBench-specific functionality for analyzing performance vs context length.
        """
        length_ranges = [(0, 4000), (4000, 8000), (8000, 16000), (16000, float('inf'))]
        results = {}
        
        for min_len, max_len in length_ranges:
            range_name = f"{min_len}-{max_len if max_len != float('inf') else 'inf'}"
            
            # Filter items in this length range
            range_indices = [
                i for i, length in enumerate(context_lengths) 
                if min_len <= length < max_len
            ]
            
            if not range_indices:
                results[range_name] = 0.0
                continue
            
            # Evaluate items in this range
            range_scores = []
            for i in range_indices:
                score = self.evaluate_response(responses[i], ground_truths[i], task_names[i])
                range_scores.append(score)
            
            results[range_name] = sum(range_scores) / len(range_scores)
        
        return results
    
    def evaluate_with_detailed_metrics(self, response: str, ground_truth: Any, task_name: str) -> Dict[str, float]:
        """Return detailed metrics for LongBench evaluation"""
        base_score = self.evaluate_response(response, ground_truth, task_name)
        
        metrics = {"overall_score": base_score}
        
        # Add task-specific metric names
        task_metrics = self.get_task_metrics(task_name)
        for metric in task_metrics:
            metrics[metric] = base_score
        
        return metrics