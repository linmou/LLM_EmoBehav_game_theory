#!/usr/bin/env python3
"""
OpenAI-Compatible Server Module Entry Point

Allows running the server as a module: python -m openai_server
"""

from .server import main

if __name__ == "__main__":
    main()
