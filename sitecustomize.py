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


def _patch_gemma3_sliding_window_pattern() -> None:
    try:
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig  # type: ignore
    except Exception:
        return

    if hasattr(Gemma3TextConfig, "sliding_window_pattern"):
        return

    def _get(self):  # type: ignore[no-untyped-def]
        return getattr(self, "_sliding_window_pattern", None)

    def _set(self, value):  # type: ignore[no-untyped-def]
        setattr(self, "_sliding_window_pattern", value)

    Gemma3TextConfig.sliding_window_pattern = property(_get, _set)  # type: ignore[attr-defined]


_patch_gemma3_sliding_window_pattern()
