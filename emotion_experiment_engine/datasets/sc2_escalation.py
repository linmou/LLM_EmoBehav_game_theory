"""StarCraft II escalation dataset adapter."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel

from ..data_models import BenchmarkItem
from .base import BaseBenchmarkDataset
from .games import GameTheoryDataset, REPO_ROOT, _DECISION_PATTERN, _SINGLE_QUOTE_PATTERN

logger = logging.getLogger(__name__)


class SC2EscalationDataset(GameTheoryDataset):
    """Benchmark dataset for StarCraft II escalation game scenarios."""

    _SCENARIO_PATH = Path("data/sc2/escalation_game.json")

    def __init__(
        self,
        config,
        prompt_wrapper: Optional[Any] = None,
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        BaseBenchmarkDataset.__init__(
            self,
            config=config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )
        self._llm_client = None

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        path = self._resolve_sc2_data_path()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError(
                f"SC2 escalation data file {path} must contain a list of scenarios"
            )

        use_full_options = bool(
            getattr(self.config, "augmentation_config", None)
            and self.config.augmentation_config.get("use_full_options")
        )

        items: List[BenchmarkItem] = []
        for idx, record in enumerate(data):
            description = str(record.get("description", "")).strip()
            if not description:
                continue

            behaviour = record.get("behaviour_decisions") or {}
            escalate_opts = behaviour.get("escalate") or []
            withdraw_opts = behaviour.get("withdraw") or []

            all_options: List[Dict[str, Any]] = []
            strengths_template = [2, 1, -1, -2]
            for text, category, strength in zip(
                list(escalate_opts) + list(withdraw_opts),
                ["escalate", "escalate", "withdraw", "withdraw"],
                strengths_template,
            ):
                all_options.append(
                    {
                        "id": len(all_options) + 1,
                        "text": str(text),
                        "category": category,
                        "escalation_strength": strength,
                    }
                )

            if not all_options:
                continue

            if use_full_options:
                options_for_prompt = all_options
            else:
                primary = [all_options[0], all_options[-1]]
                options_for_prompt = []
                for new_id, opt in enumerate(primary, start=1):
                    cloned = dict(opt)
                    cloned["id"] = new_id
                    options_for_prompt.append(cloned)

            metadata: Dict[str, Any] = {
                "options": options_for_prompt,
                "all_options": all_options,
                "you_play_as": record.get("you_play_as"),
                "players": record.get("players"),
                "behaviour_decisions": behaviour,
            }

            items.append(
                BenchmarkItem(
                    id=record.get("id", idx),
                    input_text=description,
                    context=None,
                    ground_truth=None,
                    metadata=metadata,
                )
            )

        if not items:
            raise ValueError("No SC2 escalation scenarios could be loaded")

        return items

    def _resolve_sc2_data_path(self) -> Path:
        raw_candidate = self.config.data_path or self._SCENARIO_PATH
        candidate = Path(raw_candidate)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate

        if candidate.exists():
            self.config.data_path = candidate
            return candidate

        if self.config.base_data_dir is not None:
            fallback = Path(self.config.base_data_dir) / candidate.name
            if fallback.exists():
                self.config.data_path = fallback
                return fallback

        raise FileNotFoundError(
            f"SC2 escalation data file not found for task '{self.config.task_type}'. "
            f"Expected at {candidate}"
        )

    class _ExtractionSchema(BaseModel):
        option_id: int
        rationale: str
        decision: str

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        """
        Override to avoid referencing GameTheoryDataset._match_option (protects
        against older loaded classes during runs). Returns the selected option id.
        """
        del ground_truth, task_name

        def _match_option(candidate: str, options: Sequence[str]) -> Optional[int]:
            normalized = candidate.lower().strip()
            for idx, option in enumerate(options, start=1):
                opt_norm = option.lower().strip()
                if normalized == opt_norm:
                    return idx
                
                if normalized in opt_norm or opt_norm in normalized:
                    return idx
            return None

        options = self._extract_options_from_prompt(prompt)
        candidates: List[str] = []
        for pattern in (_DECISION_PATTERN, _SINGLE_QUOTE_PATTERN):
            match = pattern.search(response)
            if match:
                candidates.append(match.group(1).strip())
        if not candidates:
            match = re.search(
                r"decision\s*[:=]\s*([^\n\r]+)", response, re.IGNORECASE
            )
            if match:
                candidates.append(match.group(1).strip())

        for candidate in candidates:
            choice_id = _match_option(candidate, options)
            if choice_id is not None:
                return float(choice_id)

        choice_id = self._fallback_option_via_llm(response, options)
        if choice_id is not None:
            return float(choice_id)

        logger.warning("Failed to extract option id for response: %s", response)
        return math.nan
