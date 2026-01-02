#!/usr/bin/env python3
# Integration test for emotion ranking output.

from pathlib import Path
import json
import numpy as np

from auto_experiments.layer_vector_sim.pd_steering_similarity import run_pd_steering_similarity


def test_emotion_ranking_output(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"

    pd_vec_dir = tmp_path / "pd_vectors"
    pd_vec_dir.mkdir(parents=True)
    np.save(pd_vec_dir / "layer_0.npy", np.array([1.0, 0.0], dtype=np.float32))

    steering_root = tmp_path / "steering_root" / "layer_vectors"
    steering_root.mkdir(parents=True)
    np.save(steering_root / "layer_0.npy", np.array([1.0, 0.0], dtype=np.float32))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model:
  name: test-model
  path: /models/test
benchmark:
  name: game_theory
  task: Prisoners_Dilemma
  raw_results_path: {fixtures / "raw_results_groups.json"}
steering:
  emotions: ["anger", "fear"]
  intensities: [1.0]
  loader: emotion_experiment_engine.experiment.EmotionExperiment
pd_defection_vectors:
  dir: {pd_vec_dir}
output:
  dir: {tmp_path / "output"}
"""
    )

    def fake_hidden_state_fn(sample_id: str, steering_condition_id: str):
        # Make anger align poorly and fear align well to test ranking order
        if steering_condition_id.startswith("anger"):
            baseline = {0: np.array([1.0, 0.0], dtype=np.float32)}
            steered = {0: np.array([0.0, 1.0], dtype=np.float32)}
        else:
            baseline = {0: np.array([0.0, 1.0], dtype=np.float32)}
            steered = {0: np.array([1.0, 0.0], dtype=np.float32)}
        return baseline, steered

    run_pd_steering_similarity.run_analysis(
        config_path=config_path,
        steering_root=steering_root,
        hidden_state_fn=fake_hidden_state_fn,
    )

    ranking_path = config_path.parent / "output" / "emotion_rankings.json"
    assert ranking_path.exists()
    rankings = json.loads(ranking_path.read_text())
    assert rankings[0]["steering_condition_id"].startswith("fear")
