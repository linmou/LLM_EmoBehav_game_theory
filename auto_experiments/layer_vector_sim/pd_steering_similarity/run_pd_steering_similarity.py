"""
Entrypoint for PD steering similarity analysis.

Usage:
  python -m auto_experiments.layer_vector_sim.pd_steering_similarity.run_pd_steering_similarity --config <config.yaml> [--steering_root <dir>]
"""

from pathlib import Path
from typing import Callable, Dict, Tuple, Optional

import numpy as np

from . import benchmark_io, config_schema, hidden_state_capture, layer_similarity
from . import output_writer, pd_defection_loader, sample_grouping, steering_loader, group_aggregation, emotion_aggregation

HiddenStateFn = Callable[[str, str], Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]]


def run_analysis(
    config_path: Path,
    steering_root: Optional[Path] = None,
    hidden_state_fn: Optional[HiddenStateFn] = None,
) -> Path:
    cfg = config_schema.load_config(Path(config_path))

    # Resolve paths relative to config file location
    config_dir = Path(config_path).parent
    raw_results_path = _resolve(config_dir, cfg.benchmark.raw_results_path)
    pd_vectors_dir = _resolve(config_dir, cfg.pd_defection_vectors.dir)
    steering_dir = _resolve(config_dir, steering_root or cfg.pd_defection_vectors.dir)

    samples = sample_grouping.load_samples(raw_results_path)
    switchers = sample_grouping.filter_switchers(samples)
    non_switchers = sample_grouping.filter_non_switchers(samples)

    pd_vectors = pd_defection_loader.load_pd_defection_vectors(pd_vectors_dir)
    steering_loader.load_emotion_vectors(steering_dir)  # ensure available; not used directly here

    hs_fn = hidden_state_fn or hidden_state_capture.get_hidden_states_for_sample

    all_records: list[layer_similarity.LayerSimilarityRecord] = []
    for sample in switchers:
        for emotion in cfg.steering.emotions:
            for intensity in cfg.steering.intensities:
                steering_condition_id = f"{emotion}_{intensity}"
                hidden_baseline, hidden_steered = hs_fn(sample.sample_id, steering_condition_id)
                records = layer_similarity.compute_similarity_records(
                    sample_id=sample.sample_id,
                    steering_condition_id=steering_condition_id,
                    hidden_baseline=hidden_baseline,
                    hidden_steered=hidden_steered,
                    pd_defection_vectors=pd_vectors,
                )
                all_records.extend(records)

    sim_path = output_writer.write_similarity_records(all_records, cfg.output.dir)

    group_labels = {s.sample_id: "switcher" for s in switchers}
    group_labels.update({s.sample_id: "non-switcher" for s in non_switchers})
    summaries = group_aggregation.aggregate_by_group(all_records, group_labels)
    output_writer.write_group_summaries(summaries, cfg.output.dir)

    rankings = emotion_aggregation.rank_emotions(
        [
            emotion_aggregation.GroupSummaryInput(
                steering_condition_id=s.steering_condition_id,
                mean_similarity_delta=s.mean_similarity_delta,
            )
            for s in summaries
            if s.group_label == "switcher"
        ]
    )
    output_writer.write_emotion_rankings(rankings, cfg.output.dir)

    return sim_path


def _resolve(base: Path, path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()
