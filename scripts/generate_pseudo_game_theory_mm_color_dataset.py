#!/usr/bin/env python3
"""Generate a tiny pseudo multimodal dataset for color recognition (JSON + images).

This dataset is meant to sanity-check whether a VLM can read a simple color cue
from an image and choose the matching option.

Output schema: list of records compatible with `game_theory_mm`:
- Uses `Prisoners_Dilemma` scenario schema (scenario/description/participants/behavior_choices)
- Adds `image_path` per record
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import Image


def generate_color_dataset(*, repo_root: Path) -> Path:
    out_dir = repo_root / "data" / "pseudo" / "game_theory_mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = out_dir / "images_color_smoke"
    img_dir.mkdir(parents=True, exist_ok=True)

    images: Dict[str, Path] = {}
    for color in ("blue", "red"):
        p = img_dir / f"{color}.png"
        Image.new("RGB", (96, 96), color=color).save(p)
        images[color] = p

    def _record(item_id: str, color: str) -> dict:
        return {
            "id": item_id,
            "scenario": "Color Recognition Smoke Test",
            "description": (
                "Look at the image. Decide which option matches the color of the square. "
                'Choose "BLUE" if the image is blue, otherwise choose "RED".'
            ),
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {"cooperate": "BLUE", "defect": "RED"},
            "image_path": str(images[color].relative_to(repo_root)),
        }

    records: List[dict] = [
        _record("mm-color-1", "blue"),
        _record("mm-color-2", "red"),
    ]

    out_json = out_dir / "pseudo_prisoners_dilemma_mm_color.json"
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return out_json


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_json = generate_color_dataset(repo_root=repo_root)
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()

