#!/usr/bin/env python3
"""
OpenAI-Compatible Server Entry Point (Backward Compatibility)

This script provides backward compatibility for the old init_openai_server.py usage.
The actual server implementation is now in the openai_server module.

New usage:
    python -m openai_server --model <path> --model_name <name> --emotion <emotion>

Old usage (still supported):
    python init_openai_server.py --model <path> --model_name <name> --emotion <emotion>
"""

import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import from the new module structure
from openai_server.server import main

def main_wrapper():
    """Wrapper to provide deprecation warning."""
    warnings.warn(
        "Using init_openai_server.py is deprecated. "
        "Please use 'python -m openai_server' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    main()

if __name__ == "__main__":
    main_wrapper()