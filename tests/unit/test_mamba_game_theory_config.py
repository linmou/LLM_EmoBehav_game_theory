# Tests config/new_game_theory_config.yaml to ensure the Mamba Prisoners Dilemma setup uses the new schema.
from pathlib import Path

import yaml


def test_mamba_prisoners_dilemma_config_matches_new_schema() -> None:
    config_path = Path("config/new_game_theory_config.yaml")
    raw = yaml.safe_load(config_path.read_text())

    assert raw.get("experiment_name") == "game_theory_mamba_pd"
    assert raw.get("version") == "1.0.0"
    assert raw.get("models") == [
        "/data/home/jjl7137/huggingface_models/state-spaces/mamba-790m-hf"
    ]
    assert raw.get("emotions") == ["anger", "happiness"]
    assert raw.get("intensities") == [1.5]

    benchmarks = raw.get("benchmarks")
    assert isinstance(benchmarks, list) and len(benchmarks) == 1
    benchmark = benchmarks[0]
    assert benchmark["name"] == "game_theory"
    assert benchmark["task_type"] == "Prisoners_Dilemma"
    assert benchmark["sample_limit"] == 36
    assert benchmark["augmentation_config"]["previous_actions_length"] == 0

    generation_cfg = raw.get("generation_config", {})
    assert generation_cfg.get("temperature") == 0.7
    assert generation_cfg.get("top_p") == 0.95
    assert generation_cfg.get("do_sample") is True
    assert generation_cfg.get("max_new_tokens") == 440

    assert raw.get("batch_size") == 50
    assert raw.get("repeat_runs") == 1
    assert raw.get("output_dir") == "results/Mamba_Series_Prisoners_Dilemma"
