"""Tests for image+text multimodal stimulus support.

Responsible files:
- neuro_manipulation/utils.py (primary_emotions_concept_dataset, detect_emotion_data_type)

Purpose:
- Ensure multimodal stimulus JSON entries can be objects containing both an image path
  and accompanying text (image+text), producing dict items with {"images": [...], "text": ...}.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def test_primary_emotions_concept_dataset_supports_image_plus_text(tmp_path: Path) -> None:
    """I am starting with a failing test. This is the Red phase."""
    data_dir = tmp_path / "stimulus"
    data_dir.mkdir()

    img_a = data_dir / "anger.jpg"
    img_h = data_dir / "happiness.jpg"
    Image.new("RGB", (8, 8), "red").save(img_a)
    Image.new("RGB", (8, 8), "yellow").save(img_h)

    (data_dir / "anger.json").write_text(
        json.dumps([{"image": "anger.jpg", "text": "Anger caption"}]),
        encoding="utf-8",
    )
    (data_dir / "happiness.json").write_text(
        json.dumps([{"image": "happiness.jpg", "text": "Happy caption"}]),
        encoding="utf-8",
    )

    from neuro_manipulation.utils import primary_emotions_concept_dataset

    data = primary_emotions_concept_dataset(
        str(data_dir),
        multimodal_intent=True,
        emotions=["anger", "happiness"],
    )

    sample = data["anger"]["train"]["data"][0]
    assert isinstance(sample, dict)
    assert sample["text"].endswith("Anger caption")
    assert len(sample["images"]) == 1
    assert isinstance(sample["images"][0], Image.Image)

