"""Responsible file: scripts/prebuild_vlm_readers.py.

Purpose: verify one-model sanity configs and GPU assignment for parallel RepE
reader prebuild runs.
"""

from pathlib import Path

import yaml

from scripts.prebuild_vlm_readers import (
    build_single_model_sanity_config,
    build_single_model_jobs,
    run_parallel_prebuilds,
)


def test_build_single_model_sanity_config_preserves_reader_inputs():
    base_config = {
        "models": ["model-a", "model-b"],
        "emotions": ["anger", "sadness"],
        "intensities": [0.8, 1.0, 1.5],
        "benchmarks": [{"name": "game_theory_decision", "task_type": "Prisoners_Dilemma"}],
        "repe_eng_config": {
            "data_dir": "multimodal_crow_envnt/emotion_envent",
            "multimodal_intent": True,
        },
        "batch_size": 800,
        "output_dir": "results/vlm_mm_game_theory_decision/sample300",
    }

    config = build_single_model_sanity_config(base_config, "model-b")

    assert config["models"] == ["model-b"]
    assert config["emotions"] == ["anger", "sadness"]
    assert config["intensities"] == [0.8, 1.0, 1.5]
    assert config["repe_eng_config"]["data_dir"] == "multimodal_crow_envnt/emotion_envent"
    assert config["sanity_check"] is True
    assert config["sanity_check_limit"] == 2
    assert config["defer_evaluation"] is True
    assert config["output_dir"].endswith("sample300_reader_prebuild")


def test_build_single_model_jobs_round_robins_gpus():
    jobs = build_single_model_jobs(
        models=["model-a", "model-b", "model-c", "model-d", "model-e"],
        gpu_ids=[0, 1, 2],
    )

    assert [job.model_path for job in jobs] == [
        "model-a",
        "model-b",
        "model-c",
        "model-d",
        "model-e",
    ]


def test_run_parallel_prebuilds_schedules_next_job_on_any_free_gpu(tmp_path, monkeypatch):
    base_config = {
        "models": ["model-a", "model-b", "model-c", "model-d", "model-e", "model-f"],
        "output_dir": "results/vlm_mm_game_theory_decision/sample300",
    }
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    launch_order: list[str] = []
    processes: list[FakeProcess] = []

    class FakeProcess:
        def __init__(self, model_path: str, polls_until_done: int) -> None:
            self.model_path = model_path
            self.polls_until_done = polls_until_done

        def poll(self) -> int | None:
            if self.polls_until_done <= 0:
                return 0
            return None

        def wait(self) -> int:
            self.polls_until_done = 0
            return 0

    def fake_launch(job, config_path_arg: Path, log_path: Path):
        del config_path_arg, log_path
        launch_order.append(job.model_path)
        process = FakeProcess(
            model_path=job.model_path,
            polls_until_done=3 if job.model_path == "model-a" else 0,
        )
        processes.append(process)
        return process

    def fake_sleep(seconds: float) -> None:
        del seconds
        for process in processes:
            if process.polls_until_done > 0:
                process.polls_until_done -= 1

    monkeypatch.setattr("scripts.prebuild_vlm_readers._launch_job", fake_launch)
    monkeypatch.setattr("scripts.prebuild_vlm_readers.time.sleep", fake_sleep)

    run_parallel_prebuilds(
        config_path=config_path,
        gpu_ids=[0, 1, 2, 3],
        work_dir=tmp_path,
        launch=True,
    )

    assert launch_order[:4] == ["model-a", "model-b", "model-c", "model-d"]
    assert launch_order[4:] == ["model-e", "model-f"]


def test_run_parallel_prebuilds_skips_models_with_completed_logs(tmp_path, monkeypatch):
    base_config = {
        "models": ["model-a", "model-b"],
        "output_dir": "results/vlm_mm_game_theory_decision/sample300",
    }
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    logs_dir = tmp_path / "logs" / "reader_prebuild"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "model-a_gpu0.log").write_text(
        "2026-03-21 INFO Memory experiment series completed. Final status: {'completed': 7}\n",
        encoding="utf-8",
    )

    launch_order: list[str] = []

    class FakeProcess:
        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    def fake_launch(job, config_path_arg: Path, log_path: Path):
        del config_path_arg, log_path
        launch_order.append(job.model_path)
        return FakeProcess()

    monkeypatch.setattr("scripts.prebuild_vlm_readers._launch_job", fake_launch)

    generated = run_parallel_prebuilds(
        config_path=config_path,
        gpu_ids=[0, 1],
        work_dir=tmp_path,
        launch=True,
    )

    assert [path.name for path in generated] == ["model-b_gpu0.yaml"]
    assert launch_order == ["model-b"]


def test_run_parallel_prebuilds_uses_any_free_gpu_for_remaining_models(tmp_path, monkeypatch):
    base_config = {
        "models": ["model-a", "model-b", "model-c", "model-d", "model-e"],
        "output_dir": "results/vlm_mm_game_theory_decision/sample300",
    }
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    logs_dir = tmp_path / "logs" / "reader_prebuild"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for name, gpu_id in [
        ("model-b", 1),
        ("model-c", 2),
        ("model-d", 3),
    ]:
        (logs_dir / f"{name}_gpu{gpu_id}.log").write_text(
            "2026-03-21 INFO Memory experiment series completed. Final status: {'completed': 7}\n",
            encoding="utf-8",
        )

    launched: list[tuple[str, str]] = []

    class FakeProcess:
        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    def fake_launch(job, config_path_arg: Path, log_path: Path):
        del log_path
        launched.append((job.model_path, config_path_arg.name))
        return FakeProcess()

    monkeypatch.setattr("scripts.prebuild_vlm_readers._launch_job", fake_launch)

    generated = run_parallel_prebuilds(
        config_path=config_path,
        gpu_ids=[0, 1, 2, 3],
        work_dir=tmp_path,
        launch=True,
    )

    assert [path.name for path in generated] == ["model-a_gpu0.yaml", "model-e_gpu1.yaml"]
    assert launched == [
        ("model-a", "model-a_gpu0.yaml"),
        ("model-e", "model-e_gpu1.yaml"),
    ]


def test_run_parallel_prebuilds_requeues_startup_timeout_to_another_gpu(tmp_path, monkeypatch):
    base_config = {
        "models": ["model-a", "model-b"],
        "output_dir": "results/vlm_mm_game_theory_decision/sample300",
    }
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    launched: list[tuple[str, str]] = []
    now = {"value": 0.0}
    active_proc_by_gpu: dict[int, FakeProcess] = {}

    class FakeProcess:
        def __init__(self, model_path: str, gpu_id: int, stuck: bool) -> None:
            self.model_path = model_path
            self.gpu_id = gpu_id
            self.stuck = stuck
            self.terminated = False

        def poll(self) -> int | None:
            if self.terminated:
                return -15
            if self.stuck:
                return None
            return 0

        def wait(self) -> int:
            self.terminated = True
            return 0

    def fake_launch(job, config_path_arg: Path, log_path: Path):
        del log_path
        gpu_id = int(config_path_arg.stem.rsplit("_gpu", 1)[1])
        launched.append((job.model_path, config_path_arg.name))
        stuck = job.model_path == "model-a" and gpu_id == 0
        process = FakeProcess(model_path=job.model_path, gpu_id=gpu_id, stuck=stuck)
        active_proc_by_gpu[gpu_id] = process
        return process

    def fake_sleep(seconds: float) -> None:
        now["value"] += seconds

    def fake_time() -> float:
        return now["value"]

    def fake_gpu_memory_used(gpu_id: int) -> int:
        process = active_proc_by_gpu.get(gpu_id)
        if process is None or process.terminated:
            return 0
        if process.stuck:
            return 4
        return 4096

    def fake_terminate_job(job) -> None:
        job.process.terminated = True

    monkeypatch.setattr("scripts.prebuild_vlm_readers._launch_job", fake_launch)
    monkeypatch.setattr("scripts.prebuild_vlm_readers.time.sleep", fake_sleep)
    monkeypatch.setattr("scripts.prebuild_vlm_readers.time.time", fake_time)
    monkeypatch.setattr("scripts.prebuild_vlm_readers._gpu_memory_used_mib", fake_gpu_memory_used)
    monkeypatch.setattr("scripts.prebuild_vlm_readers._terminate_running_job", fake_terminate_job)

    run_parallel_prebuilds(
        config_path=config_path,
        gpu_ids=[0, 1],
        work_dir=tmp_path,
        launch=True,
        startup_timeout_seconds=5.0,
    )

    assert launched == [
        ("model-a", "model-a_gpu0.yaml"),
        ("model-b", "model-b_gpu1.yaml"),
        ("model-a", "model-a_gpu1.yaml"),
    ]


def test_run_parallel_prebuilds_requeues_progress_timeout_to_another_gpu(tmp_path, monkeypatch):
    base_config = {
        "models": ["model-a", "model-b"],
        "output_dir": "results/vlm_mm_game_theory_decision/sample300",
    }
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    launched: list[tuple[str, str]] = []
    now = {"value": 0.0}
    active_proc_by_gpu: dict[int, FakeProcess] = {}
    log_mtime_by_name: dict[str, float] = {}

    class FakeProcess:
        def __init__(self, model_path: str, gpu_id: int, stalled: bool) -> None:
            self.model_path = model_path
            self.gpu_id = gpu_id
            self.stalled = stalled
            self.terminated = False

        def poll(self) -> int | None:
            if self.terminated:
                return -15
            if self.stalled:
                return None
            return 0

        def wait(self) -> int:
            self.terminated = True
            return 0

    def fake_launch(job, config_path_arg: Path, log_path: Path):
        gpu_id = int(config_path_arg.stem.rsplit("_gpu", 1)[1])
        launched.append((job.model_path, config_path_arg.name))
        stalled = job.model_path == "model-a" and gpu_id == 0
        process = FakeProcess(model_path=job.model_path, gpu_id=gpu_id, stalled=stalled)
        active_proc_by_gpu[gpu_id] = process
        log_mtime_by_name[log_path.name] = now["value"]
        return process

    def fake_sleep(seconds: float) -> None:
        now["value"] += seconds

    def fake_time() -> float:
        return now["value"]

    def fake_gpu_memory_used(gpu_id: int) -> int:
        process = active_proc_by_gpu.get(gpu_id)
        if process is None or process.terminated:
            return 0
        return 4096

    def fake_log_mtime(path: Path) -> float:
        return log_mtime_by_name[path.name]

    def fake_terminate_job(job) -> None:
        job.process.terminated = True

    monkeypatch.setattr("scripts.prebuild_vlm_readers._launch_job", fake_launch)
    monkeypatch.setattr("scripts.prebuild_vlm_readers.time.sleep", fake_sleep)
    monkeypatch.setattr("scripts.prebuild_vlm_readers.time.time", fake_time)
    monkeypatch.setattr("scripts.prebuild_vlm_readers._gpu_memory_used_mib", fake_gpu_memory_used)
    monkeypatch.setattr("scripts.prebuild_vlm_readers._log_mtime_seconds", fake_log_mtime)
    monkeypatch.setattr("scripts.prebuild_vlm_readers._terminate_running_job", fake_terminate_job)

    run_parallel_prebuilds(
        config_path=config_path,
        gpu_ids=[0, 1],
        work_dir=tmp_path,
        launch=True,
        startup_timeout_seconds=999.0,
        progress_timeout_seconds=5.0,
    )

    assert launched == [
        ("model-a", "model-a_gpu0.yaml"),
        ("model-b", "model-b_gpu1.yaml"),
        ("model-a", "model-a_gpu1.yaml"),
    ]
