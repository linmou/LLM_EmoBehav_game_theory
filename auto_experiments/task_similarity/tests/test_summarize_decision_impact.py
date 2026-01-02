"""
Responsible: auto_experiments/task_similarity/summarize_similarity_decision_impact.py
Purpose: Validate core summary helpers (top correlations + last-layer slice).
"""

import pandas as pd


def test_select_last_layers():
    from auto_experiments.task_similarity.summarize_similarity_decision_impact import select_last_layers

    df = pd.DataFrame(
        {
            "intensity": [0.6] * 6,
            "layer": [30, 31, 32, 33, 34, 35],
            "pearson_r(defect,cosine)": [0, 1, 2, 3, 4, 5],
        }
    )
    out = select_last_layers(df, k=5)
    assert out["layer"].tolist() == [31, 32, 33, 34, 35]


def test_top_abs_pearson_per_intensity():
    from auto_experiments.task_similarity.summarize_similarity_decision_impact import top_abs_pearson_per_intensity

    df = pd.DataFrame(
        {
            "intensity": [0.6, 0.6, 0.8, 0.8],
            "layer": [1, 2, 1, 2],
            "pearson_r(defect,cosine)": [0.1, -0.3, 0.02, -0.01],
        }
    )
    out = top_abs_pearson_per_intensity(df, top_k=1)
    # best abs is layer 2 for 0.6 (-0.3), and layer 1 for 0.8 (0.02)
    rows = {(float(r.intensity), int(r.layer)) for r in out.itertuples(index=False)}
    assert rows == {(0.6, 2), (0.8, 1)}

