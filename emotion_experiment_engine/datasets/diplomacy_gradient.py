"""
Dataset for Diplomacy PD-style gradient decisions (1..5 options).

Input file format (JSONL): list of objects with at least
 - id: str
 - scenario: str (includes consequence framing in prose)
 - options: List[{id:int, text:str}] with 1..5 entries (natural-language orders)

This dataset intentionally avoids dependency on games/game_configs and uses
BaseBenchmarkDataset common loader utilities.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

from ..data_models import BenchmarkItem
from .base import BaseBenchmarkDataset


_OPTION_LINE_PATTERN = re.compile(r"\s*Option\s*(\d+)[\.:\)]\s*(.+)", re.IGNORECASE)


class DiplomacyGradientDataset(BaseBenchmarkDataset):
    """Simple dataset adapter for PD-style Diplomacy choices."""

    LLM_EVAL_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "client": "openai",
    }

    # ------------------------- Data loading -------------------------
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        raw = self._load_raw_data()
        items: List[BenchmarkItem] = []
        for i, rec in enumerate(raw):
            scenario = rec.get("scenario") or rec.get("event") or ""
            options = rec.get("options") or []
            if not scenario or not options:
                # Skip malformed rows silently to keep dataset robust
                continue
            norm_options: List[Dict[str, Any]] = []
            for j, opt in enumerate(options):
                if isinstance(opt, dict):
                    text = opt.get("text") or opt.get("value") or str(opt)
                    opt_id = opt.get("id") or j + 1
                else:
                    text = str(opt)
                    opt_id = j + 1
                norm_options.append({"id": int(opt_id), "text": str(text)})

            # Build background context header (Your Country + Game + Phase + Target)
            your_country = rec.get("your_country")
            target_country = rec.get("target_country")
            game_name = rec.get("game")
            phase = rec.get("phase") or {}
            year = phase.get("year")
            season = phase.get("season")
            subphase = phase.get("subphase")
            header_lines = []
            if your_country:
                header_lines.append(f"Your Country: {your_country}")
            if game_name:
                header_lines.append(f"Game: {game_name}")
            if season or year or subphase:
                # Compose a compact phase line
                phase_bits = []
                if season: phase_bits.append(str(season))
                if year: phase_bits.append(str(year))
                if subphase: phase_bits.append(str(subphase))
                header_lines.append("Phase: " + " ".join(phase_bits))
            if target_country:
                header_lines.append(f"Target Country: {target_country}")
            context_header = "\n".join(header_lines) if header_lines else None

            items.append(
                BenchmarkItem(
                    id=rec.get("id", i),
                    input_text=str(scenario),
                    context=context_header,
                    ground_truth=None,
                    metadata={"options": norm_options},
                )
            )

        if not items:
            raise ValueError("No valid Diplomacy PD-style items found")
        return items

    # ------------------------- Evaluation -------------------------
    @staticmethod
    def _extract_options_from_prompt(prompt: str) -> List[str]:
        opts: List[str] = []
        for line in prompt.splitlines():
            m = _OPTION_LINE_PATTERN.match(line)
            if m:
                opts.append(m.group(2).strip())
        return opts

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip()).lower()

    def _extract_option_from_response(
        self, response: str, options: Sequence[str]
    ) -> Optional[int]:
        # Prefer explicit numeric form if present
        m = re.search(r"option\s*(\d+)", response, flags=re.IGNORECASE)
        if m:
            try:
                k = int(m.group(1))
                if 1 <= k <= len(options):
                    return k
            except Exception:
                pass

        # Otherwise match by option text (case/space-insensitive)
        resp = self._normalize(response)
        for idx, text in enumerate(options, start=1):
            if self._normalize(text) in resp or resp in self._normalize(text):
                return idx
        return None

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        del ground_truth, task_name
        options = self._extract_options_from_prompt(prompt)
        choice = self._extract_option_from_response(response or "", options)
        return float(choice) if choice is not None else math.nan

    def get_task_metrics(self, task_name: str) -> List[str]:
        del task_name
        return ["option_id"]


__all__ = ["DiplomacyGradientDataset"]
