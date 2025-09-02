"""
MTBench101 data preparation script - TDD Phase 1 Green
Minimal implementation to make tests pass.
Purpose: Split mtbench101.jsonl into task-specific files
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Expected task distribution from actual MTBench101 data
# Based on analysis: AR=153, IC=150, SI=149, CC=147, CR=136, MR=108, etc.
EXPECTED_TASK_COUNTS = {
    "AR": 153,  # Anaphora Resolution
    "IC": 150,  # Interrogative Completion  
    "SI": 149,  # Stepwise Instruction
    "CC": 147,  # Context Carryover
    "CR": 136,  # Content Rewriting
    "MR": 108,  # Mathematical Reasoning
    "PI": 87,   # Proactive Interaction
    "TS": 83,   # Topic Switching
    "CM": 80,   # Conversational Memory
    "SC": 77,   # Self-Correction
    "FR": 74,   # Format Revision
    "SA": 73,   # Self-Affirmation
    "GR": 71,   # General Reasoning
}


def _validate_conversation(conversation: Dict[str, Any], line_num: int) -> None:
    """Validate required fields in conversation object."""
    required_fields = ["task", "id", "history"]
    for field in required_fields:
        if field not in conversation:
            raise ValueError(f"Missing '{field}' field at line {line_num}")


def split_mtbench101_by_task(input_file: str, output_dir: Path) -> None:
    """
    Split MTBench101 data by task type into separate files.
    
    Args:
        input_file: Path to input mtbench101.jsonl file
        output_dir: Directory to write task-specific files
        
    Raises:
        json.JSONDecodeError: If input contains malformed JSON
        FileNotFoundError: If input file doesn't exist
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group conversations by task
    conversations_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
                
            try:
                conversation = json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_num}: {e.msg}", 
                    e.doc, 
                    e.pos
                )
            
            _validate_conversation(conversation, line_num)
            task = conversation["task"]
            conversations_by_task[task].append(conversation)
    
    # Write task-specific files
    for task, conversations in conversations_by_task.items():
        output_file = output_dir / f"mtbench101_{task}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                json.dump(conversation, f, ensure_ascii=False)
                f.write('\n')


def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Split MTBench101 data by task")
    parser.add_argument("input_file", help="Input mtbench101.jsonl file")
    parser.add_argument("output_dir", help="Output directory for task files")
    
    args = parser.parse_args()
    
    split_mtbench101_by_task(args.input_file, Path(args.output_dir))
    print(f"Successfully split MTBench101 data into {args.output_dir}")


if __name__ == "__main__":
    main()