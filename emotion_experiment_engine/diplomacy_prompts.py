"""
DiplomacyOptionsPromptWrapper

Prompts for Diplomacy PD-style gradient decisions.

Adds explicit background (Your Country, Game, Phase, Target Country) before the
scenario and renders natural-language options 1..5.

Subclass of `neuro_manipulation.prompt_wrapper.PromptWrapper` so it composes
cleanly in BenchmarkSpec.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from neuro_manipulation.prompt_wrapper import PromptWrapper


class DiplomacyOptionsPromptWrapper(PromptWrapper):
    def __init__(self, prompt_format: Any | None) -> None:
        super().__init__(prompt_format)

    @staticmethod
    def _normalize_options(options: Optional[Sequence[Any]]) -> List[str]:
        if not options:
            return []
        out: List[str] = []
        for opt in options:
            if isinstance(opt, dict):
                text = opt.get("text") or opt.get("value")
                out.append(str(text) if text is not None else str(opt))
            else:
                out.append(str(opt))
        return out

    @staticmethod
    def _render_header(context: Optional[str]) -> str:
        if not context:
            return ""
        ctx = str(context).strip()
        return ctx + ("\n\n" if ctx else "")

    @staticmethod
    def _render_event_options(event: str, options: Sequence[str]) -> str:
        lines = [str(event).strip()]
        for i, opt in enumerate(options, start=1):
            lines.append(f"Option {i}. {opt}")
        lines.append("Respond with the option text.")
        return "\n".join(lines)

    def __call__(
        self,
        *,
        context: str | None,
        question: str,
        user_messages: Sequence[str] | str | None,
        enable_thinking: bool,
        augmentation_config: Optional[dict],  # unused, adapter compatibility
        answer: Any,  # unused
        emotion: Optional[str],  # unused
        options: Optional[Sequence[Any]],
    ) -> str:
        del augmentation_config, answer, emotion
        normalized = self._normalize_options(options)

        # Normalize user messages
        if user_messages is None:
            user_messages_list: List[str] = ["Please provide your answer."]
        elif isinstance(user_messages, str):
            user_messages_list = [user_messages]
        else:
            user_messages_list = list(user_messages)

        header = self._render_header(context)
        body = self._render_event_options(question, normalized)
        prompt_text = header + body

        # Use PromptFormat.build if available; otherwise return plain text
        build_fn = getattr(self.prompt_format, "build", None)
        if callable(build_fn):
            return build_fn(prompt_text, user_messages_list, enable_thinking=enable_thinking)
        return prompt_text


__all__ = ["DiplomacyOptionsPromptWrapper"]

