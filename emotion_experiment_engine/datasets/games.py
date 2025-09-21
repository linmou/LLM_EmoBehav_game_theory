"""Dataset adapter for game theory benchmarks."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from games.game import SequentialGameScenario
from games.game_configs import get_game_config
from neuro_manipulation.utils import oai_response
from pydantic import BaseModel

from ..data_models import BenchmarkItem, ResultRecord
from .base import BaseBenchmarkDataset

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
_DECISION_PATTERN = re.compile(r'"decision"\s*[:=]\s*"([^"]+)"', re.IGNORECASE)
_SINGLE_QUOTE_PATTERN = re.compile(r"'decision'\s*[:=]\s*'([^']+)'", re.IGNORECASE)
_OPTION_LINE_PATTERN = re.compile(r"\s*Option\s*(\d+)[\.:\)]\s*(.+)", re.IGNORECASE)


class GameTheoryDataset(BaseBenchmarkDataset):
    """Benchmark dataset that exposes game theory scenarios as BenchmarkItems."""

    LLM_EVAL_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "client": "openai",
    }

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
        base_config = deepcopy(get_game_config(config.task_type))
        if config.augmentation_config:
            base_config.update(config.augmentation_config)
        self._game_config = base_config
        super().__init__(
            config=config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
            **kwargs,
        )
        self._llm_client = None  # Lazily constructed for fallback parsing

    # ---------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        raw_items = self._load_raw_scenarios()

        # if "scenario_class" not in self._game_config: # remove so that error can be raised
        #     return self._build_items_from_raw(raw_items)

        scenario_class = self._game_config["scenario_class"]
        payoff_matrix = self._game_config["payoff_matrix"]
        augmentation = self.config.augmentation_config or {}
        scenario_fields = getattr(scenario_class, "model_fields", {})
        config_fields = self._game_config

        items: List[BenchmarkItem] = []
        for idx, record in enumerate(raw_items):
            enriched = dict(record)
            if "payoff_matrix" not in enriched:
                enriched["payoff_matrix"] = payoff_matrix

            for field_name in scenario_fields:
                if field_name in augmentation and field_name not in enriched:
                    enriched[field_name] = augmentation[field_name]
                elif field_name in config_fields and field_name not in enriched:
                    enriched[field_name] = config_fields[field_name]

            try:
                scenario = scenario_class(**enriched)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Skipping scenario %s due to parse error: %r", idx, exc
                )
                continue

            options = [
                {"id": opt_idx + 1, "text": choice}
                for opt_idx, choice in enumerate(
                    scenario.get_behavior_choices().get_choices()
                )
            ]

            item_id = enriched.get("id", idx)
            metadata: Dict[str, Any] = {
                "options": options,
            }

            if isinstance(scenario, SequentialGameScenario):
                previous_attr = getattr(scenario, "previous_actions", None)
                try:
                    resolved = previous_attr() if callable(previous_attr) else previous_attr
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to resolve previous actions for scenario %s: %r",
                        item_id,
                        exc,
                    )
                    resolved = None

                if resolved:
                    previous_list = list(resolved)
                    metadata["previous_actions"] = previous_list
                    metadata["previous_actions_length"] = len(previous_list)

            items.append(
                BenchmarkItem(
                    id=item_id,
                    input_text=str(scenario),
                    context=None,
                    ground_truth=None,
                    metadata=metadata,
                )
            )

        if not items:
            raise ValueError(
                f"No scenarios could be loaded for task '{self.config.task_type}'"
            )

        return items

    def _build_items_from_raw(
        self, raw_items: Sequence[Dict[str, Any]]
    ) -> List[BenchmarkItem]:

        items: List[BenchmarkItem] = []
        for idx, record in enumerate(raw_items):
            event = record.get("event") or record.get("scenario") or ""
            option_entries = record.get("options") or []

            normalized_options: List[Dict[str, Any]] = []
            for opt_idx, opt in enumerate(option_entries):
                if isinstance(opt, dict):
                    text = opt.get("text") or opt.get("value") or str(opt)
                    opt_id = opt.get("id") or opt_idx + 1
                else:
                    text = str(opt)
                    opt_id = opt_idx + 1
                normalized_options.append({"id": opt_id, "text": text})

            items.append(
                BenchmarkItem(
                    id=record.get("id", idx),
                    input_text=str(event),
                    context=None,
                    ground_truth=None,
                    metadata={"options": normalized_options},
                )
            )

        if not items:
            raise ValueError("Raw scenario list was empty")

        return items

    class _ExtractionSchema(BaseModel):
        option_id: int
        rationale: str
        decision: str

    def _resolve_data_path(self) -> Path:
        candidate = Path(self._game_config["data_path"])
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
            f"Game data file not found for task '{self.config.task_type}'. "
            f"Expected at {candidate}"
        )

    def _load_raw_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = self._game_config.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            return [dict(item) for item in scenarios]

        path = self._resolve_data_path()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError(
                f"Game data file {path} must contain a list of scenarios"
            )
        return data

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        del ground_truth, task_name  # Accuracy not meaningful; return option id

        options = self._extract_options_from_prompt(prompt)
        choice_id = self._extract_option_from_response(response, options)

        if choice_id is not None:
            return float(choice_id)

        choice_id = self._fallback_option_via_llm(response, options)
        if choice_id is not None:
            return float(choice_id)

        logger.warning("Failed to extract option id for response: %s", response)
        return math.nan

    def get_task_metrics(self, task_name: str) -> List[str]:
        del task_name
        return ["option_id"]

    def compute_split_metrics(self, records: List[ResultRecord]) -> Dict[str, Any]:
        base_metrics = super().compute_split_metrics(records)
        overall_rows = self._choice_ratio_rows(records, include_repeat=False)
        repeat_rows = self._choice_ratio_rows(records, include_repeat=True)

        if not overall_rows and not repeat_rows:
            return base_metrics

        metrics = dict(base_metrics) if isinstance(base_metrics, dict) else {}
        metrics["choice_ratio"] = {
            "overall": overall_rows,
            "by_repeat": repeat_rows,
        }
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_options_from_prompt(prompt: str) -> List[str]:
        options: List[str] = []
        for line in prompt.splitlines():
            match = _OPTION_LINE_PATTERN.match(line)
            if match:
                options.append(match.group(2).strip())
        return options

    @staticmethod
    def _extract_option_from_response(
        response: str, options: Sequence[str]
    ) -> Optional[int]:
        candidates = []
        for pattern in (_DECISION_PATTERN, _SINGLE_QUOTE_PATTERN):
            match = pattern.search(response)
            if match:
                candidates.append(match.group(1).strip())
        if not candidates:
            # Handle bare "decision: value" cases
            match = re.search(
                r"decision\s*[:=]\s*([^\n\r]+)", response, re.IGNORECASE
            )
            if match:
                candidates.append(match.group(1).strip())

        for candidate in candidates:
            option_id = GameTheoryDataset._match_option(candidate, options)
            if option_id is not None:
                return option_id
        return None

    @staticmethod
    def _match_option(candidate: str, options: Sequence[str]) -> Optional[int]:
        normalized = candidate.lower().strip()
        for idx, option in enumerate(options, start=1):
            opt_norm = option.lower().strip()
            if normalized == opt_norm:
                return idx
        for idx, option in enumerate(options, start=1):
            opt_norm = option.lower().strip()
            if normalized in opt_norm or opt_norm in normalized:
                return idx
        return None

    def _choice_ratio_rows(
        self, records: List[ResultRecord], *, include_repeat: bool
    ) -> List[Dict[str, Any]]:
        option_counts: Dict[Tuple[Any, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        total_counts: Dict[Tuple[Any, ...], int] = defaultdict(int)

        for record in records:
            score = record.score
            if score is None:
                continue

            try:
                option_val = float(score)
            except (TypeError, ValueError):
                continue

            if math.isnan(option_val):
                continue

            option_id = int(option_val)
            key_parts: List[Any] = [record.emotion, record.intensity]
            if include_repeat:
                key_parts.append(record.repeat_id)

            key = tuple(key_parts)
            option_counts[key][option_id] += 1
            total_counts[key] += 1

        rows: List[Dict[str, Any]] = []
        for key in sorted(option_counts.keys()):
            total = total_counts[key]
            if not total:
                continue

            for option_id in sorted(option_counts[key].keys()):
                row = {
                    "emotion": key[0],
                    "intensity": key[1],
                    "option_id": option_id,
                    "ratio": option_counts[key][option_id] / total,
                }
                if include_repeat:
                    row["repeat_id"] = key[2]
                rows.append(row)

        return rows

    def _fallback_option_via_llm(
        self, response: str, options: Sequence[str]
    ) -> Optional[int]:
        if not options:
            return None

        client = self._ensure_llm_client()
        if client is None:
            return None

        formatted_options = ", ".join(
            f"Option {idx + 1}: {text}" for idx, text in enumerate(options)
        )
        prompt = (
            "You are helping classify a model's decision. Given the available options "
            f"({formatted_options}), identify which option best matches the following "
            f"response. Respond with JSON containing an integer field named option_id.\n\n"
            f"Response:\n{response}"
        )

        try:
            result = oai_response(
                prompt,
                client=client,
                model=self.llm_eval_config.get("model", "gpt-4o-mini"),
                response_format=self._ExtractionSchema,
            )
        except Exception as exc:  # pragma: no cover - network failure safeguard
            logger.warning("LLM extraction failed: %s", exc)
            return None

        return self._parse_option_id_from_result(result)

    @staticmethod
    def _parse_option_id_from_result(result: Any) -> Optional[int]:
        if isinstance(result, BaseModel):
            option_id = getattr(result, "option_id", None)
            if isinstance(option_id, int) and option_id > 0:
                return option_id
            return None

        if isinstance(result, dict):
            option_id = result.get("option_id")
            if isinstance(option_id, int) and option_id > 0:
                return option_id
            if isinstance(option_id, str) and option_id.isdigit():
                return int(option_id)
            return None

        text = str(result)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                option_id = data.get("option_id")
                if isinstance(option_id, int) and option_id > 0:
                    return option_id
                if isinstance(option_id, str) and option_id.isdigit():
                    return int(option_id)
        except json.JSONDecodeError:
            pass

        match = re.search(r"option_id\s*[:=]\s*([0-9]+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"option\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _ensure_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client

        client_name = str(self.llm_eval_config.get("client", "openai")).lower()
        try:
            if client_name == "azure":
                from openai import AzureOpenAI  # type: ignore

                from api_configs import AZURE_OPENAI_CONFIG

                self._llm_client = AzureOpenAI(**AZURE_OPENAI_CONFIG)
            else:
                from openai import OpenAI  # type: ignore

                from api_configs import OAI_CONFIG

                self._llm_client = OpenAI(**OAI_CONFIG)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Unable to initialise LLM client: %s", exc)
            self._llm_client = None
        return self._llm_client


__all__ = ["GameTheoryDataset"]
