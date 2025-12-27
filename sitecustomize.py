"""
Repo-local Python startup hooks.

KISS: vLLM currently expects `Gemma3TextConfig.sliding_window_pattern`, but some
transformers builds expose only `_sliding_window_pattern`. Provide a tiny alias
property so vLLM can read it.
"""

from __future__ import annotations


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
