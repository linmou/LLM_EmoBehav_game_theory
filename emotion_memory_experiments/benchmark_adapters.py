"""
Benchmark adapters for different memory benchmark suites.
Each adapter handles loading data and creating prompts for specific benchmarks.
"""
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add InfiniteBench to path for evaluation functions
sys.path.append('/data/home/jjl7137/memory_benchmarks/InfiniteBench/src')

try:
    from compute_scores import get_score_one
except ImportError:
    print("Warning: Could not import InfiniteBench compute_scores. Some functionality may be limited.")
    get_score_one = None

from .data_models import BenchmarkItem, BenchmarkConfig


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark adapters"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data = None
    
    @abstractmethod
    def load_data(self) -> List[BenchmarkItem]:
        """Load and parse benchmark data into standardized format"""
        pass
    
    @abstractmethod
    def create_prompt(self, item: BenchmarkItem) -> str:
        """Create a prompt string from a benchmark item"""
        pass
    
    @abstractmethod
    def evaluate_response(self, response: str, ground_truth: Any, task_name: str) -> float:
        """Evaluate a response against ground truth using benchmark-specific method"""
        pass
    
    def get_data(self) -> List[BenchmarkItem]:
        """Get loaded data, loading if necessary"""
        if self.data is None:
            self.data = self.load_data()
        return self.data


class InfiniteBenchAdapter(BenchmarkAdapter):
    """Adapter for InfiniteBench tasks"""
    
    def load_data(self) -> List[BenchmarkItem]:
        """Load InfiniteBench data from JSONL format"""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")
        
        items = []
        
        if data_path.suffix == '.jsonl':
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        item_data = json.loads(line)
                        items.append(self._parse_infinitebench_item(item_data, i))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                for i, item_data in enumerate(data_list):
                    items.append(self._parse_infinitebench_item(item_data, i))
        
        if self.config.sample_limit:
            items = items[:self.config.sample_limit]
        
        return items
    
    def _parse_infinitebench_item(self, item_data: Dict[str, Any], index: int) -> BenchmarkItem:
        """Parse a single InfiniteBench item into standard format"""
        # InfiniteBench format varies by task, try common fields
        item_id = item_data.get('id', index)
        
        # Handle different input formats
        input_text = item_data.get('input', '')
        context = item_data.get('context', '')
        
        # Some tasks combine context and input
        if not input_text and context:
            input_text = context
            context = None
        
        # Ground truth can be in different fields
        ground_truth = item_data.get('answer', 
                                   item_data.get('label', 
                                               item_data.get('ground_truth', None)))
        
        return BenchmarkItem(
            id=item_id,
            input_text=input_text,
            context=context,
            ground_truth=ground_truth,
            metadata=item_data
        )
    
    def create_prompt(self, item: BenchmarkItem) -> str:
        """Create prompt for InfiniteBench tasks"""
        if item.context:
            # Task with separate context and question
            return f"{item.context}\n\nQuestion: {item.input_text}\nAnswer:"
        else:
            # Question only
            return f"{item.input_text}\nAnswer:"
    
    def evaluate_response(self, response: str, ground_truth: Any, task_name: str) -> float:
        """Evaluate using InfiniteBench's evaluation functions"""
        if get_score_one is None:
            # Fallback evaluation - exact match
            return 1.0 if str(response).strip().lower() == str(ground_truth).strip().lower() else 0.0
        
        try:
            score = get_score_one(response, ground_truth, task_name, "emotion_model")
            return float(score)
        except Exception as e:
            print(f"Evaluation error for {task_name}: {e}")
            return 0.0


class LoCoMoAdapter(BenchmarkAdapter):
    """Adapter for LoCoMo conversational memory benchmark"""
    
    def load_data(self) -> List[BenchmarkItem]:
        """Load LoCoMo data"""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        for sample in data:
            sample_id = sample.get('sample_id', len(items))
            conversation = sample.get('conversation', {})
            qa_pairs = sample.get('qa', [])
            
            for qa in qa_pairs:
                items.append(BenchmarkItem(
                    id=f"{sample_id}_{len(items)}",
                    input_text=qa['question'],
                    context=self._format_conversation(conversation),
                    ground_truth=qa['answer'],
                    metadata={
                        'sample_id': sample_id,
                        'category': qa.get('category', 'unknown'),
                        'evidence': qa.get('evidence', [])
                    }
                ))
        
        if self.config.sample_limit:
            items = items[:self.config.sample_limit]
        
        return items
    
    def _format_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format LoCoMo conversation data into context string"""
        formatted_sessions = []
        
        # Extract sessions in chronological order
        session_keys = [k for k in conversation.keys() if k.startswith('session_') and not k.endswith('_date_time')]
        session_keys.sort()
        
        for session_key in session_keys:
            session_data = conversation[session_key]
            date_key = f"{session_key}_date_time"
            session_date = conversation.get(date_key, "Unknown date")
            
            formatted_sessions.append(f"=== {session_key.upper()} ({session_date}) ===")
            
            for turn in session_data:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                formatted_sessions.append(f"{speaker}: {text}")
            
            formatted_sessions.append("")  # Empty line between sessions
        
        return "\n".join(formatted_sessions)
    
    def create_prompt(self, item: BenchmarkItem) -> str:
        """Create prompt for LoCoMo conversational QA"""
        return f"{item.context}\n\nQuestion: {item.input_text}\nAnswer:"
    
    def evaluate_response(self, response: str, ground_truth: Any, task_name: str) -> float:
        """Evaluate LoCoMo responses using simple similarity"""
        # For now, use simple token overlap (could be enhanced with ROUGE/BLEU)
        response_tokens = set(response.lower().split())
        ground_truth_tokens = set(str(ground_truth).lower().split())
        
        if not ground_truth_tokens:
            return 0.0
        
        overlap = len(response_tokens & ground_truth_tokens)
        return overlap / len(ground_truth_tokens)


def get_adapter(config: BenchmarkConfig) -> BenchmarkAdapter:
    """Factory function to get appropriate adapter for benchmark"""
    if config.name.lower() in ['infinitebench', 'infinite_bench']:
        return InfiniteBenchAdapter(config)
    elif config.name.lower() in ['locomo', 'loco_mo']:
        return LoCoMoAdapter(config)
    else:
        raise ValueError(f"Unknown benchmark: {config.name}")