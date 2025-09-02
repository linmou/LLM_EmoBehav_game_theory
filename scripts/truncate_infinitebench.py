#!/usr/bin/env python3
"""
Script to truncate InfiniteBench LongBook QA data to 121k tokens.
Uses the _apply_truncation method from BaseBenchmarkDataset.
"""

import json
import sys
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from emotion_memory_experiments.data_models import BenchmarkItem
from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset


class InfiniteBenchTruncator(BaseBenchmarkDataset):
    """Minimal implementation to use _apply_truncation method"""
    
    def __init__(self, tokenizer, max_context_length: int = 121000):
        # Skip parent __init__ to avoid loading data
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.truncation_strategy = "right"  # Keep beginning of context
        
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        # Not used in this script
        return []
        
    def evaluate_response(self, response: str, ground_truth, task_name: str) -> float:
        # Not used in this script
        return 0.0
        
    def get_task_metrics(self, task_name: str) -> List[str]:
        # Not used in this script
        return []


def load_jsonl(file_path: Path) -> List[dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[dict], file_path: Path):
    """Save data as JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_to_benchmark_items(raw_data: List[dict]) -> List[BenchmarkItem]:
    """Convert InfiniteBench format to BenchmarkItem"""
    items = []
    for item in raw_data:
        benchmark_item = BenchmarkItem(
            id=item["id"],
            context=item.get("context", ""),
            input_text=item["input"],
            ground_truth=item["answer"],
            metadata={
                "options": item.get("options", [])
            }
        )
        items.append(benchmark_item)
    return items


def convert_back_to_original_format(items: List[BenchmarkItem]) -> List[dict]:
    """Convert BenchmarkItem back to InfiniteBench format"""
    result = []
    for item in items:
        original_format = {
            "id": item.id,
            "context": item.context or "",
            "input": item.input_text,
            "answer": item.ground_truth,
            "options": item.metadata.get("options", []) if item.metadata else []
        }
        result.append(original_format)
    return result


def main():
    # Initialize tokenizer - try Qwen first, fallback to GPT-2
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        print("Using Qwen2.5-0.5B tokenizer")
    except Exception as e:
        print(f"Failed to load Qwen tokenizer: {e}")
        print("Falling back to GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT-2 doesn't have a pad token, so we'll add one
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Create truncator
    truncator = InfiniteBenchTruncator(tokenizer, max_context_length=121000)
    
    # Process both files
    input_files = [
        "data/memory_benchmarks/infinitebench_longbook_qa_chn.jsonl",
        "data/memory_benchmarks/infinitebench_longbook_qa_eng.jsonl"
    ]
    
    for input_file in input_files:
        input_path = Path(input_file)
        output_path = input_path.parent / f"{input_path.stem}_121k{input_path.suffix}"
        
        print(f"\nProcessing {input_path}...")
        
        # Load data
        raw_data = load_jsonl(input_path)
        print(f"Loaded {len(raw_data)} items")
        
        # Convert to BenchmarkItem format
        benchmark_items = convert_to_benchmark_items(raw_data)
        
        # Apply truncation using the batch method
        print("Applying truncation...")
        truncated_items = truncator._apply_truncation(benchmark_items)
        
        # Convert back to original format
        output_data = convert_back_to_original_format(truncated_items)
        
        # Save truncated data
        save_jsonl(output_data, output_path)
        print(f"Saved truncated data to {output_path}")
        
        # Print truncation statistics
        truncated_count = sum(
            1 for item in truncated_items 
            if item.metadata and item.metadata.get("truncation_info", {}).get("was_truncated", False)
        )
        print(f"Truncated {truncated_count}/{len(truncated_items)} items")


if __name__ == "__main__":
    main()