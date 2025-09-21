"""Prompt wrapper adapter for game theory benchmarks."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from games.game_configs import get_game_config
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper


class GameBenchmarkPromptWrapper:
    """Adapts game theory prompts to the benchmark wrapper signature."""

    def __init__(self, prompt_format: Any | None, task_type: str) -> None:
        self.prompt_format = prompt_format
        self.task_type = task_type
        self._config = get_game_config(task_type)
        self._decision_class = self._config["decision_class"]
        self._react_wrapper: Optional[GameReactPromptWrapper] = None

        if self.prompt_format is not None:
            self._react_wrapper = GameReactPromptWrapper(
                self.prompt_format, self._decision_class
            )

    def _ensure_wrapper(self) -> GameReactPromptWrapper | None:
        if self._react_wrapper is None and self.prompt_format is not None:
            self._react_wrapper = GameReactPromptWrapper(
                self.prompt_format, self._decision_class
            )
        return self._react_wrapper

    @staticmethod
    def _normalize_options(options: Optional[Sequence[Any]]) -> List[str]:
        if not options:
            return []
        normalized: List[str] = []
        for opt in options:
            if isinstance(opt, dict):
                text = opt.get("text") or opt.get("value")
                if text is None:
                    text = str({k: v for k, v in opt.items() if k != "id"})
                normalized.append(str(text))
            else:
                normalized.append(str(opt))
        return normalized

    @staticmethod
    def _fallback_prompt(event: str, options: Sequence[str]) -> str:
        prompt_lines = [event]
        for idx, option in enumerate(options, start=1):
            prompt_lines.append(f"Option {idx}. {option}")
        prompt_lines.append("Respond with the option text.")
        return "\n".join(prompt_lines)

    def __call__(
        self,
        *,
        context: str | None,
        question: str,
        user_messages: Sequence[str] | str | None,
        enable_thinking: bool,
        augmentation_config: Optional[dict],
        answer: Any,
        emotion: Optional[str],
        options: Optional[Sequence[Any]],
    ) -> str:
        del context, augmentation_config, answer, emotion  # unused in adapter
        normalized_options = self._normalize_options(options)

        wrapper = self._ensure_wrapper()
        if wrapper is None:
            return self._fallback_prompt(question, normalized_options)

        if user_messages is None:
            user_messages = ["Please provide your answer."]
        elif isinstance(user_messages, str):
            user_messages = [user_messages]

        prompt_text = wrapper(
            event=question,
            options=normalized_options,
            user_messages=list(user_messages),
            enable_thinking=enable_thinking,
        )

        if not prompt_text:
            return self._fallback_prompt(question, normalized_options)

        return prompt_text


__all__ = ["GameBenchmarkPromptWrapper"]
