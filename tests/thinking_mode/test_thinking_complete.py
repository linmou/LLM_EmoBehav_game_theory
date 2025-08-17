#!/usr/bin/env python3
"""
Complete test for Qwen3 thinking mode using multiple disable methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from neuro_manipulation.prompt_formats import PromptFormat

def generate_response(model, tokenizer, prompt, max_tokens=300):
    """Generate response using the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (response)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def test_thinking_methods():
    """Test different methods to control thinking mode"""
    
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen3-1.7B"
    
    print("=" * 80)
    print("QWEN3 THINKING MODE CONTROL TEST")
    print("=" * 80)
    
    # Load model and tokenizer
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    prompt_format = PromptFormat(tokenizer)
    
    # Test question
    system_prompt = "You are Alice. Answer the question directly."
    user_question = "Should I choose Option 1 (cooperate) or Option 2 (defect)? Give me a brief answer."
    
    print("\n" + "=" * 60)
    print("METHOD 1: enable_thinking=False (API level)")
    print("=" * 60)
    
    # Method 1: Use enable_thinking=False in prompt format
    prompt1 = prompt_format.build(
        system_prompt=system_prompt,
        user_messages=[user_question],
        assistant_messages=[],
        enable_thinking=False
    )
    
    print("Prompt contains '/think':", "/think" in prompt1)
    print("Generated prompt:")
    print(prompt1)
    print("\nGenerating response...")
    
    response1 = generate_response(model, tokenizer, prompt1)
    print("\nRESPONSE (enable_thinking=False):")
    print("-" * 50)
    print(response1)
    print("-" * 50)
    
    has_think1 = "<think>" in response1
    print(f"Response contains <think>: {has_think1}")
    
    print("\n" + "=" * 60)
    print("METHOD 2: /no_think directive")
    print("=" * 60)
    
    # Method 2: Use /no_think directive
    user_with_directive = "/no_think\n" + user_question
    prompt2 = prompt_format.build(
        system_prompt=system_prompt,
        user_messages=[user_with_directive],
        assistant_messages=[],
        enable_thinking=True  # Enable thinking but use directive to disable
    )
    
    print("Prompt contains '/no_think':", "/no_think" in prompt2)
    print("Generated prompt:")
    print(prompt2)
    print("\nGenerating response...")
    
    response2 = generate_response(model, tokenizer, prompt2)
    print("\nRESPONSE (/no_think directive):")
    print("-" * 50)
    print(response2)
    print("-" * 50)
    
    has_think2 = "<think>" in response2
    print(f"Response contains <think>: {has_think2}")
    
    print("\n" + "=" * 60)
    print("METHOD 3: System message with /no_think")
    print("=" * 60)
    
    # Method 3: Add /no_think to system message
    system_no_think = system_prompt + " /no_think"
    prompt3 = prompt_format.build(
        system_prompt=system_no_think,
        user_messages=[user_question],
        assistant_messages=[],
        enable_thinking=True
    )
    
    print("System prompt contains '/no_think':", "/no_think" in system_no_think)
    print("Generated prompt:")
    print(prompt3)
    print("\nGenerating response...")
    
    response3 = generate_response(model, tokenizer, prompt3)
    print("\nRESPONSE (system /no_think):")
    print("-" * 50)
    print(response3)
    print("-" * 50)
    
    has_think3 = "<think>" in response3
    print(f"Response contains <think>: {has_think3}")
    
    print("\n" + "=" * 60)
    print("METHOD 4: Baseline (thinking enabled)")
    print("=" * 60)
    
    # Method 4: Baseline with thinking enabled
    prompt4 = prompt_format.build(
        system_prompt=system_prompt,
        user_messages=[user_question],
        assistant_messages=[],
        enable_thinking=True
    )
    
    print("Generated prompt:")
    print(prompt4)
    print("\nGenerating response...")
    
    response4 = generate_response(model, tokenizer, prompt4)
    print("\nRESPONSE (thinking enabled):")
    print("-" * 50)
    print(response4)
    print("-" * 50)
    
    has_think4 = "<think>" in response4
    print(f"Response contains <think>: {has_think4}")
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    methods = [
        ("enable_thinking=False", has_think1),
        ("/no_think directive", has_think2),
        ("system /no_think", has_think3),
        ("thinking enabled", has_think4)
    ]
    
    print("Method                  | Has <think> | Expected")
    print("-" * 50)
    for method, has_think in methods:
        expected = "❌" if method != "thinking enabled" else "✅"
        result = "❌" if has_think else "✅"
        if method == "thinking enabled":
            expected = "✅" if has_think else "❌"
            result = "✅" if has_think else "❌"
        print(f"{method:20} | {result:8} | {expected}")
    
    # Check which methods successfully disabled thinking
    successful_methods = []
    for method, has_think in methods[:-1]:  # Exclude baseline
        if not has_think:
            successful_methods.append(method)
    
    if successful_methods:
        print(f"\n✅ SUCCESS: These methods disabled thinking: {', '.join(successful_methods)}")
    else:
        print("\n❌ FAILURE: No method successfully disabled thinking mode")
    
    # Verify baseline works
    if has_think4:
        print("✅ Baseline confirmed: Thinking mode works when enabled")
    else:
        print("❌ Baseline issue: Thinking mode not working even when enabled")

if __name__ == "__main__":
    test_thinking_methods()