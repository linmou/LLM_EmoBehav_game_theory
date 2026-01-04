"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Ensure similarity run directories follow the <datetime>/model/<emotion>/seed_<seed> layout.
"""

from pathlib import Path


def test_build_output_dir_layout():
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import build_output_dir

    out = build_output_dir(
        output_root=Path("auto_experiments/task_similarity/results/emotion_pd_delta_similarity"),
        run_id="20260102_135547",
        model_path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        emotion="anger",
        split_seed=20,
    )
    assert out.as_posix().endswith(
        "auto_experiments/task_similarity/results/emotion_pd_delta_similarity/20260102_135547/"
        "Qwen2.5-0.5B-Instruct/anger/seed_20"
    )
