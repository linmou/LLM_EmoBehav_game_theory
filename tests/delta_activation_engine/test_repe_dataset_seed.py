"""
Responsible: tests/delta_activation_engine/test_repe_dataset_seed.py
Purpose: Ensure RepE dataset seed propagates so delta activations can vary splits.
"""

def test_load_emotion_readers_uses_emotion_data_seed(monkeypatch):
    import neuro_manipulation.model_utils as mu
    import neuro_manipulation.utils as nm_utils

    seeds = []
    hashed_args = []

    monkeypatch.setattr(
        nm_utils,
        "validate_multimodal_experiment_feasibility",
        lambda _cfg: {"feasible": True, "mode": "text", "reasons": []},
    )

    def fake_dataset(
        data_dir,
        model_name=None,
        tokenizer=None,
        system_prompt=None,
        seed=0,
        multimodal_intent=False,
        enable_thinking=False,
    ):
        seeds.append(seed)
        return {
            "anger": {
                "train": {"data": [], "labels": []},
                "test": {"data": [], "labels": []},
            }
        }

    def fake_dict_to_unique_code(args):
        hashed_args.append(args)
        return "seed-hash"

    def fake_all_emotion_rep_reader(
        data,
        emotions,
        rep_reading_pipeline,
        hidden_layers,
        rep_token,
        n_difference,
        direction_method,
        read_args,
        save_path,
    ):
        return {"args": read_args}

    class DummyPipeline:
        def __call__(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(mu, "primary_emotions_concept_dataset", fake_dataset)
    monkeypatch.setattr(mu, "dict_to_unique_code", fake_dict_to_unique_code)
    monkeypatch.setattr(mu, "all_emotion_rep_reader", fake_all_emotion_rep_reader)
    monkeypatch.setattr(mu, "pipeline", lambda *args, **kwargs: DummyPipeline())

    cfg = {
        "emotions": ["anger"],
        "data_dir": "data/stimulus/text/",
        "model_name_or_path": "dummy",
        "rep_token": -1,
        "n_difference": 1,
        "direction_method": "pca",
        "emotion_data_seed": 13,
    }

    readers = mu.load_emotion_readers(cfg, model=object(), tokenizer=object(), hidden_layers=[0])

    assert seeds == [13]
    assert hashed_args and hashed_args[0]["emotion_data_seed"] == 13
    assert readers["args"]["emotion_data_seed"] == 13
