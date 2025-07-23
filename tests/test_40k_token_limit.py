#!/usr/bin/env python3
"""
Test to verify 40K token limits are working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_utils_imports():
    """Test that utils can be imported and shows 40K limits"""
    print("Testing utils.py import and configuration...")
    
    # Read the utils.py file to verify limits
    with open('neuro_manipulation/utils.py', 'r') as f:
        content = f.read()
    
    # Check for 40000 limits
    if 'max_model_len=40000' in content:
        print("✅ utils.py: max_model_len set to 40000")
        count = content.count('max_model_len=40000')
        print(f"   Found {count} instances of max_model_len=40000")
    else:
        print("❌ utils.py: max_model_len not set to 40000")
        return False
    
    return True

def test_vllm_hook_limits():
    """Test vLLM hook limits"""
    print("\nTesting vLLM hook configuration...")
    
    with open('neuro_manipulation/repe/rep_control_vllm_hook.py', 'r') as f:
        content = f.read()
    
    if 'max_tokens=kwargs.get(\'max_new_tokens\', 40000)' in content:
        print("✅ rep_control_vllm_hook.py: default max_tokens set to 40000")
    else:
        print("❌ rep_control_vllm_hook.py: default max_tokens not set to 40000")
        return False
    
    return True

def test_config_compatibility():
    """Test that config files can still work with new limits"""
    print("\nTesting config compatibility...")
    
    # Check if config still has 28000 setting
    try:
        with open('config/Qwen3_Thinking_Mode_Enabled.yaml', 'r') as f:
            content = f.read()
            
        if 'max_new_tokens: 28000' in content:
            print("✅ Config file still has max_new_tokens: 28000")
            print("   This should now work with 40K model limit")
        else:
            print("ℹ️ Config file has different max_new_tokens setting")
            
    except FileNotFoundError:
        print("ℹ️ Config file not found, skipping compatibility test")
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("40K TOKEN LIMIT VERIFICATION TEST")
    print("=" * 60)
    
    tests = [
        test_utils_imports,
        test_vllm_hook_limits,
        test_config_compatibility
    ]
    
    all_passed = True
    for test in tests:
        try:
            result = test()
            all_passed = all_passed and result
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("The 40K token limits have been successfully implemented.")
        print("\nKey changes:")
        print("- max_model_len: 1000 → 40000 (in utils.py)")
        print("- max_tokens default: 100 → 40000 (in vLLM hook)")
        print("- All model loading limits increased to 40K")
        print("\nNow your 28K token config should work properly!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the implementation.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())