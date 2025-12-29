"""
sitecustomize.py

Python automatically imports `sitecustomize` at interpreter startup (if it is
importable via sys.path). vLLM worker processes use `spawn`, so this is the
most reliable place to apply tiny, opt-in runtime tweaks via environment vars.
"""

from __future__ import annotations

import os

_VLLM_DISABLE_FLASH_ATTN_APPLIED = False

if os.environ.get("VLLM_DISABLE_FLASH_ATTN", "0") == "1":
    try:
        import transformers.utils as _tutils
        import transformers.utils.import_utils as _iu

        _tutils.is_flash_attn_2_available = lambda: False  # type: ignore[assignment]
        _iu.is_flash_attn_2_available = lambda: False  # type: ignore[assignment]
        _VLLM_DISABLE_FLASH_ATTN_APPLIED = True
    except Exception:
        # Best effort: do not crash interpreter startup.
        _VLLM_DISABLE_FLASH_ATTN_APPLIED = False

