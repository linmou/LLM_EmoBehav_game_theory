#!/usr/bin/env python3
"""Check if qwen_vl_utils is available."""

try:
    import qwen_vl_utils
    print('✓ qwen_vl_utils is available')
    print(f'Module location: {qwen_vl_utils.__file__}')
    print(f'Available functions: {dir(qwen_vl_utils)}')
except ImportError as e:
    print('❌ qwen_vl_utils not found')
    print(f'Error: {e}')
    print('You may need to install it with: pip install qwen-vl-utils')