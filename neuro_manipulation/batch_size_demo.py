#!/usr/bin/env python
"""
Demo script showing how to use the BatchSizeFinder API to find optimal batch sizes.
"""
import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuro_manipulation.gpu_optimization import (
    BatchSizeFinder,
    find_optimal_batch_size_for_llm,
    measure_throughput
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Demo for BatchSizeFinder API")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        help="HuggingFace model name")
    parser.add_argument("--sample-text", type=str, default="Say a story as long as you can.", 
                        help="Sample text for testing")
    parser.add_argument("--max-length", type=int, default=128, 
                        help="Max sequence length")
    parser.add_argument("--safety-margin", type=float, default=0.9, 
                        help="Memory safety margin (0.9 = 90%)")
    parser.add_argument("--mode", type=str, choices=["power", "binsearch"], default="binsearch",
                        help="Search algorithm to use")
    parser.add_argument("--method", type=str, choices=["direct", "utility", "both"], default="both",
                        help="Which method to demonstrate")
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"Loading model: {args.model}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
     
    # Method 1: Use the utility function (simpler)
    if args.method in ["utility", "both"]:
        logger.info("\n=== Method 1: Using the utility function ===")
        optimal_batch_size = find_optimal_batch_size_for_llm(
            model=model,
            tokenizer=tokenizer,
            sample_text=args.sample_text,
            max_length=args.max_length,
            mode=args.mode,
            safety_margin=args.safety_margin,
            generation_kwargs={"max_new_tokens": 200}
        )
        
        logger.info(f"Optimal batch size (utility): {optimal_batch_size}")
        
        # Measure throughput with the found batch size
        throughput = measure_throughput(
            model=model,
            tokenizer=tokenizer,
            sample_text=args.sample_text,
            max_length=args.max_length,
            batch_size=optimal_batch_size
        )
        
        logger.info(f"Throughput at batch size {optimal_batch_size}: {throughput:.2f} samples/second")
    
    # Method 2: Use the BatchSizeFinder class directly (more control)
    if args.method in ["direct", "both"]:
        logger.info("\n=== Method 2: Using the BatchSizeFinder class directly ===")
        
        # Create a sample input function that takes a batch size and returns model inputs
        def sample_input_fn(batch_size):
            return tokenizer(
                [args.sample_text] * batch_size,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(model.device)
        
        # Create and use the finder
        finder = BatchSizeFinder(
            mode=args.mode,
            safety_margin=args.safety_margin,
            max_trials=15
        )
        
        optimal_batch_size = finder.find(
            model=model,
            sample_input_fn=sample_input_fn,
            generation_kwargs={"max_new_tokens": 20}
        )
        
        logger.info(f"Optimal batch size (direct): {optimal_batch_size}")
        
        # Try various batch sizes to compare memory usage and throughput
        logger.info("\n=== Comparing batch sizes ===")
        batch_sizes_to_test = [1, optimal_batch_size // 2, optimal_batch_size, min(optimal_batch_size * 2, 64)]
        
        for bs in batch_sizes_to_test:
            if bs <= 0 or bs > 128:  # Skip invalid or too large batch sizes
                continue
                
            # Measure throughput
            throughput = measure_throughput(
                model=model,
                tokenizer=tokenizer,
                sample_text=args.sample_text,
                max_length=args.max_length,
                batch_size=bs
            )
            
            logger.info(f"Batch size {bs}: {throughput:.2f} samples/second")
    
    logger.info("\nDone! You can use these batch sizes in your experiments.")

if __name__ == "__main__":
    main() 