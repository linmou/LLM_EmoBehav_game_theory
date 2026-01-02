#!/usr/bin/env python3
# Integration test for run_pd_steering_similarity pipeline with fixtures.

from pathlib import Path
import numpy as np

from auto_experiments.layer_vector_sim.pd_steering_similarity import run_pd_steering_similarity


def test_run_pipeline_writes_records(tmp_path: Path) -> None:
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
  raw_results_path: {fixtures / "raw_results_switchers.json"}
steering:
  emotions: ["anger"]
  intensities: [1.0]
  loader: emotion_experiment_engine.experiment.EmotionExperiment
pd_defection_vectors:
  dir: {pd_vec_dir}
output:
  dir: {tmp_path / "output"}
"""
    )

    def fake_hidden_state_fn(sample_id: str, steering_condition_id: str):
        return (
            {0: np.array([1.0, 0.0], dtype=np.float32)},
            {0: np.array([0.0, 1.0], dtype=np.float32)},
        )

    out_path = run_pd_steering_similarity.run_analysis(
        config_path=config_path,
        steering_root=steering_root,
        hidden_state_fn=fake_hidden_state_fn,
    )

    assert out_path.exists()
    content = out_path.read_text()
    assert "sample_id" in content
