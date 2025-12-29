"""Tests for `sitecustomize.py` runtime knobs.

Responsible files:
- sitecustomize.py

Purpose:
- Ensure `VLLM_DISABLE_FLASH_ATTN=1` triggers the patch in a fresh interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_sitecustomize_disables_flash_attn_env_flag() -> None:
    """I am starting with a failing test. This is the Red phase."""
    env = os.environ.copy()
    env["VLLM_DISABLE_FLASH_ATTN"] = "1"
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import sitecustomize; print(getattr(sitecustomize, '_VLLM_DISABLE_FLASH_ATTN_APPLIED', None))",
        ],
        env=env,
        text=True,
    )
    assert out.strip() == "True"

