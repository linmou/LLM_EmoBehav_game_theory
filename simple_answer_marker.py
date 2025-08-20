#!/usr/bin/env python3
"""
Simple Answer Marker - Add prefix/suffix to answers in benchmark datasets

Just does simple string replacement. No overengineering.
"""

import json
from typing import Union, List


def mark_answers(input_file: str, output_file: str, prefix: str = "[ANSWER]", suffix: str = "[/ANSWER]") -> None:
    """
    Add prefix/suffix markers around answers in benchmark data.
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path  
        prefix: Text to add before answer (default: "[ANSWER]")
        suffix: Text to add after answer (default: "[/ANSWER]")
    """
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        total = 0
        marked = 0
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                context = data.get('context', '')
                
                # Get answer - handle different formats
                answer = None
                if 'answers' in data and data['answers']:
                    answer = data['answers'][0]  # LongBench format
                elif 'answer' in data:
                    answer = data['answer']
                    if isinstance(answer, list) and answer:
                        answer = answer[0]  # InfiniteBench format
                
                # Mark answer in context
                if answer and context and answer in context:
                    marked_context = context.replace(answer, f"{prefix}{answer}{suffix}")
                    data['context'] = marked_context
                    marked += 1
                
                # Write result
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                total += 1
                
            except:
                continue
        
        print(f"✅ Processed {total} examples, marked {marked} answers")


# Simple usage functions
def add_brackets(input_file: str, output_file: str):
    """Add [ANSWER]...[/ANSWER] brackets"""
    mark_answers(input_file, output_file, "[ANSWER]", "[/ANSWER]")


def add_stars(input_file: str, output_file: str):
    """Add ***...*** stars"""  
    mark_answers(input_file, output_file, "***", "***")


def add_custom(input_file: str, output_file: str, prefix: str, suffix: str):
    """Add custom prefix/suffix"""
    mark_answers(input_file, output_file, prefix, suffix)


if __name__ == "__main__":
    # Test with actual data
    print("Testing simple answer marking...")
    
    # Test passkey
    add_brackets("test_data/real_benchmarks/infinitebench_passkey.jsonl", "test_passkey_marked.jsonl")
    
    # Test passage retrieval  
    add_custom("test_data/real_benchmarks/longbench_passage_retrieval_en.jsonl", 
               "test_passage_marked.jsonl", ">>>", "<<<")
    
    print("Done! Check test_*_marked.jsonl files")