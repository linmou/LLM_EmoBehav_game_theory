"""
Responsible: auto_experiments/task_similarity/migrate_emotion_pd_delta_similarity_results.py
Purpose: Ensure legacy results are mapped to the new <run_id>/<model>/<emotion>/seed_<seed> layout.
"""

from pathlib import Path


def test_is_legacy_metadata_path_and_dest_mapping(tmp_path: Path):
    from auto_experiments.task_similarity.migrate_emotion_pd_delta_similarity_results import (
        is_legacy_metadata_path,
        map_legacy_metadata_to_dest_seed_dir,
    )

    root = tmp_path / "results" / "anger_pd_delta_similarity"
    meta = (
        root
        / "Qwen2.5-0.5B-Instruct"
        / "anger"
        / "20260102_135547"
        / "seed_20"
        / "metadata.json"
    )
    assert is_legacy_metadata_path(meta, root=root)

    dest = map_legacy_metadata_to_dest_seed_dir(meta, root=root)
    assert dest.as_posix().endswith(
        "results/anger_pd_delta_similarity/20260102_135547/Qwen2.5-0.5B-Instruct/anger/seed_20"
    )


def test_new_layout_is_not_legacy(tmp_path: Path):
    from auto_experiments.task_similarity.migrate_emotion_pd_delta_similarity_results import is_legacy_metadata_path

    root = tmp_path / "results" / "anger_pd_delta_similarity"
    meta = root / "20260102_135547" / "Qwen2.5-0.5B-Instruct" / "anger" / "seed_20" / "metadata.json"
    assert not is_legacy_metadata_path(meta, root=root)
