"""
Responsible: auto_experiments/task_similarity/group_emotion_pd_delta_similarity_runs.py
Purpose: Test time-based grouping of per-emotion runs into bash-level run_id sessions.
"""

from pathlib import Path


def test_group_run_ids_by_time_window():
    from auto_experiments.task_similarity.group_emotion_pd_delta_similarity_runs import group_run_ids_by_time_window

    run_ids = [
        "20260102_134554",
        "20260102_134753",
        "20260102_134951",
        "20260102_135149",
        "20260102_135348",
        "20260102_135547",
        "20260102_160000",
        "20260102_160400",
    ]
    groups = group_run_ids_by_time_window(run_ids, window_seconds=30 * 60)
    assert groups == [
        [
            "20260102_134554",
            "20260102_134753",
            "20260102_134951",
            "20260102_135149",
            "20260102_135348",
            "20260102_135547",
        ],
        ["20260102_160000", "20260102_160400"],
    ]


def test_group_events_starts_new_session_on_anger():
    from auto_experiments.task_similarity.group_emotion_pd_delta_similarity_runs import (
        group_events_into_sessions,
        SeedEvent,
    )

    events = [
        SeedEvent(run_id="20260102_062714", model="M", emotion="sadness", seed="seed_20", seed_dir=Path("/x")),
        SeedEvent(run_id="20260102_063500", model="M", emotion="anger", seed="seed_20", seed_dir=Path("/y")),
        SeedEvent(run_id="20260102_063740", model="M", emotion="sadness", seed="seed_20", seed_dir=Path("/z")),
    ]
    sessions = group_events_into_sessions(events, window_seconds=10 * 60, start_emotion="anger")
    assert [[e.run_id for e in s] for s in sessions] == [
        ["20260102_062714"],
        ["20260102_063500", "20260102_063740"],
    ]


def test_rewrite_metadata_run_id(tmp_path: Path):
    from auto_experiments.task_similarity.group_emotion_pd_delta_similarity_runs import rewrite_metadata_run_id

    meta_path = tmp_path / "metadata.json"
    meta_path.write_text('{"run_id":"20260102_135149","emotion":"anger"}', encoding="utf-8")

    rewrite_metadata_run_id(meta_path, new_run_id="20260102_134554")
    payload = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "20260102_134554"
    assert payload["original_run_id"] == "20260102_135149"
