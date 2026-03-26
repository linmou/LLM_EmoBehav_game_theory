#!/usr/bin/env python3
# Responsible file: scripts/series_runner_watchdog.py
# Purpose: verify the generic watchdog only restarts stalled shard runs when report progress is frozen and GPUs stay idle.

from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "series_runner_watchdog.py"
_SPEC = importlib.util.spec_from_file_location("series_runner_watchdog", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_should_restart_when_signature_is_stale_and_gpus_are_idle() -> None:
    should_restart = _MODULE._should_restart

    assert should_restart(
        previous_signature=(9, 1, ("exp_a",)),
        current_signature=(9, 1, ("exp_a",)),
        last_progress_ts=1000.0,
        now_ts=1705.0,
        gpu_utils=[0.0, 0.0],
        idle_util_threshold=5.0,
        stall_seconds=600.0,
    )


def test_should_not_restart_when_report_is_advancing() -> None:
    should_restart = _MODULE._should_restart

    assert not should_restart(
        previous_signature=(9, 1, ("exp_a",)),
        current_signature=(10, 1, ("exp_b",)),
        last_progress_ts=1000.0,
        now_ts=1705.0,
        gpu_utils=[0.0, 0.0],
        idle_util_threshold=5.0,
        stall_seconds=600.0,
    )


def test_should_not_restart_when_gpus_are_still_busy() -> None:
    should_restart = _MODULE._should_restart

    assert not should_restart(
        previous_signature=(9, 1, ("exp_a",)),
        current_signature=(9, 1, ("exp_a",)),
        last_progress_ts=1000.0,
        now_ts=1705.0,
        gpu_utils=[42.0, 55.0],
        idle_util_threshold=5.0,
        stall_seconds=600.0,
    )


def test_should_restart_when_one_tp_gpu_is_dead_for_long_time() -> None:
    should_restart = _MODULE._should_restart

    assert should_restart(
        previous_signature=(9, 1, ("exp_a",)),
        current_signature=(9, 1, ("exp_a",)),
        last_progress_ts=1000.0,
        now_ts=2305.0,
        gpu_utils=[58.0, 0.0],
        idle_util_threshold=5.0,
        stall_seconds=600.0,
    )


def test_resolve_gpu_ids_maps_local_visible_ids_to_parent_cuda_visible_devices() -> None:
    resolve_gpu_ids = _MODULE._resolve_gpu_ids

    assert resolve_gpu_ids("0", inherited_cuda_visible_devices="2,3") == "2"
    assert resolve_gpu_ids("1", inherited_cuda_visible_devices="2,3") == "3"
    assert resolve_gpu_ids("0,1", inherited_cuda_visible_devices="2,3") == "2,3"


def test_resolve_gpu_ids_keeps_requested_ids_when_parent_mapping_is_not_applicable() -> None:
    resolve_gpu_ids = _MODULE._resolve_gpu_ids

    assert resolve_gpu_ids("2,3", inherited_cuda_visible_devices="2,3") == "2,3"
    assert resolve_gpu_ids("2", inherited_cuda_visible_devices=None) == "2"


def test_resolve_gpu_ids_keeps_explicit_physical_ids_under_narrowed_parent_mask() -> None:
    resolve_gpu_ids = _MODULE._resolve_gpu_ids

    assert resolve_gpu_ids("2,3", inherited_cuda_visible_devices="1,2,3,4") == "2,3"
