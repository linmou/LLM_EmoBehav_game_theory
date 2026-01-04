"""
Responsible: auto_experiments/task_similarity/run_emotion_pd_similarity_pipeline.py
Purpose: Ensure the PD similarity pipeline orchestration helpers build the correct CLI args.
"""

from __future__ import annotations

from pathlib import Path
import subprocess


def test_build_extra_pd_args_includes_tensor_flags_when_enabled() -> None:
    from auto_experiments.task_similarity.run_emotion_pd_similarity_pipeline import (
        build_extra_pd_args,
    )

    args = build_extra_pd_args(
        emotion_rep_reader=Path("/tmp/rr.pkl"),
        pd_vectors_dir=Path("/tmp/layer_vectors"),
        split_manifest=Path("/tmp/split_manifest.json"),
        save_tensors=True,
        tensor_dtype="float16",
    )

    assert "--emotion_rep_reader" in args
    assert "--save_tensors" in args
    assert "--tensor_dtype" in args


def test_build_extra_pd_args_excludes_tensor_flags_when_disabled() -> None:
    from auto_experiments.task_similarity.run_emotion_pd_similarity_pipeline import (
        build_extra_pd_args,
    )

    args = build_extra_pd_args(
        emotion_rep_reader=Path("/tmp/rr.pkl"),
        pd_vectors_dir=None,
        split_manifest=None,
        save_tensors=False,
        tensor_dtype="float16",
    )

    assert "--emotion_rep_reader" in args
    assert "--save_tensors" not in args
    assert "--tensor_dtype" not in args


def test_resolve_model_path_maps_huggingface_models_prefix() -> None:
    from auto_experiments.task_similarity.run_emotion_pd_similarity_pipeline import (
        resolve_model_path,
    )

    original = "/some/other/fs/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
    expected = Path("/data/local/hf/Qwen/Qwen2.5-0.5B-Instruct")

    def is_dir(path: Path) -> bool:
        return path == expected

    got = resolve_model_path(
        model_path=original,
        override=None,
        huggingface_root=Path("/data/local/hf"),
        is_dir=is_dir,
    )
    assert got == expected


def test_repreader_cache_path_uses_hashed_filename_and_requires_exists(tmp_path: Path) -> None:
    from auto_experiments.task_similarity.run_emotion_pd_similarity_pipeline import (
        repreader_cache_path_from_args,
        repreader_cache_path_from_repe_config,
    )

    def fake_code(_: object) -> str:
        return "ABCDEFGHIJKL"

    expected = tmp_path / "emotion_rep_reader_ABCDEFGHIJ.pkl"
    try:
        repreader_cache_path_from_args(args={"x": 1}, base_dir=tmp_path, dict_to_unique_code=fake_code)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass

    expected.write_bytes(b"ok")
    got = repreader_cache_path_from_args(args={"x": 1}, base_dir=tmp_path, dict_to_unique_code=fake_code)
    assert got == expected

    repe_cfg = {
        "emotions": ["anger"],
        "data_dir": "data/stimulus/crowd-enVent_textlike",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "rep_token": -1,
        "n_difference": 1,
        "direction_method": "pca",
        "multimodal_intent": False,
    }

    got2 = repreader_cache_path_from_repe_config(
        repe_cfg,
        num_layers=2,
        base_dir=tmp_path,
        dict_to_unique_code=fake_code,
        validate_multimodal=lambda cfg: {"feasible": True, "mode": "text_only", "reasons": []},
    )
    assert got2 == expected


def test_run_pipeline_uses_similarity_stdout_as_run_dir(tmp_path: Path, monkeypatch) -> None:
    """
    Responsible: auto_experiments/task_similarity/run_emotion_pd_similarity_pipeline.py
    Purpose: Guard run_pipeline wiring: similarity module must run and its stdout must be parsed as a path.
    """

    from auto_experiments.task_similarity import run_emotion_pd_similarity_pipeline as mod

    result_dir = tmp_path / "emotion_experiment"
    result_dir.mkdir()
    (result_dir / "experiment_config.json").write_text(
        """
        {
          "model_path": "/models/Qwen2.5-0.5B-Instruct",
          "emotions": ["anger"],
          "intensities": [1.5],
          "repe_eng_config": {
            "emotions": ["anger"],
            "data_dir": "data/stimulus/crowd-enVent_textlike",
            "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "rep_token": -1,
            "n_difference": 1,
            "direction_method": "pca",
            "multimodal_intent": false,
            "emotion_data_seed": 0
          }
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    (result_dir / "raw_results.json").write_text("[]\n", encoding="utf-8")

    rr = tmp_path / "emotion_rep_reader.pkl"
    rr.write_bytes(b"ok")

    pd_vectors_dir = tmp_path / "pd_vectors"
    pd_vectors_dir.mkdir()
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mod, "DEFAULT_OUTPUT_ROOT", tmp_path / "out")

    sim_dir = tmp_path / "sim_run"
    sim_dir.mkdir()

    calls: list[tuple[str, list[str]]] = []

    def fake_run_module(module: str, args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append((module, list(args)))
        if module == "auto_experiments.task_similarity.emotion_pd_delta_similarity":
            return subprocess.CompletedProcess(args=["python", "-m", module], returncode=0, stdout=str(sim_dir) + "\n")
        return subprocess.CompletedProcess(args=["python", "-m", module], returncode=0, stdout="")

    monkeypatch.setattr(mod, "_run_module", fake_run_module)
    monkeypatch.setattr(mod, "_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(mod, "resolve_model_path", lambda **kwargs: Path("/models/Qwen2.5-0.5B-Instruct"))

    args = mod.PipelineArgs(
        result_dir=result_dir,
        model_path_override=None,
        max_length=32,
        batch_size=2,
        device_map="auto",
        split="all",
        run_id="20260103_000000",
        pd_vectors_dir=pd_vectors_dir,
        split_manifest=split_manifest,
        emotion_rep_reader=rr,
        save_tensors=True,
        tensor_dtype="float16",
    )

    out = mod.run_pipeline(args)
    assert out.is_dir()
    assert calls and calls[0][0] == "auto_experiments.task_similarity.emotion_pd_delta_similarity"
    sim_args = calls[0][1]
    assert "--output_root" in sim_args
    i = sim_args.index("--output_root")
    assert sim_args[i + 1] == str(mod.DEFAULT_OUTPUT_ROOT)
    assert "--raw_results_path" in sim_args
    j = sim_args.index("--raw_results_path")
    assert sim_args[j + 1] == str(result_dir / "raw_results.json")
