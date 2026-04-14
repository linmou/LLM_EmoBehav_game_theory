#!/usr/bin/env python3
"""
Generate a tiny pseudo multimodal game-theory dataset (JSON + images).

Output schema (list of records):
- scenario, description, participants, behavior_choices: compatible with existing game scenarios
- image_path: path (relative to repo root) to an image file

This is for quickly testing `GameTheoryMultimodalDataset` without touching real data.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "pseudo" / "game_theory_mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for i, color in enumerate(["blue", "red"], start=1):
        p = img_dir / f"img_{i}.png"
        Image.new("RGB", (64, 64), color=color).save(p)
        img_paths.append(p)

    records = [
        {
            "id": "mm-1",
            "scenario": "Bandwidth Brinkmanship (image-conditioned)",
            "description": "Decide whether to upgrade now or delay; use the image as additional context.",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Immediately upgrade infrastructure",
                "defect": "Delay upgrade infrastructure",
            },
            "image_path": str(img_paths[0].relative_to(repo_root)),
        },
        {
            "id": "mm-2",
            "scenario": "Second scenario (image-conditioned)",
            "description": "Same game, different image.",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Immediately upgrade infrastructure",
                "defect": "Delay upgrade infrastructure",
            },
            "image_path": str(img_paths[1].relative_to(repo_root)),
        },
    ]

    out_json = out_dir / "pseudo_prisoners_dilemma_mm.json"
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print("Next (dataset smoke test):")
    print(
        "  python -c \"from pathlib import Path; "
        "from emotion_experiment_engine.data_models import BenchmarkConfig; "
        "from emotion_experiment_engine.datasets.games_multimodal import GameTheoryMultimodalDataset; "
        "cfg=BenchmarkConfig(name='game_theory_mm',task_type='Prisoners_Dilemma',data_path=Path('data/pseudo/game_theory_mm/pseudo_prisoners_dilemma_mm.json'),base_data_dir=None,sample_limit=1,augmentation_config=None,enable_auto_truncation=False,truncation_strategy='right',preserve_ratio=1.0,llm_eval_config=None); "
        "ds=GameTheoryMultimodalDataset(cfg,prompt_wrapper=None,answer_wrapper=None); "
        "print(ds[0]['images'][0].size, ds[0]['item'].metadata['image_path'])\""
    )


if __name__ == "__main__":
    main()

