"""
Responsible: rep_control_vllm_hook.py
Purpose: validate the capture-enabled hook stores pre/post activations and
supports collection via the worker RPC helper.
"""

import types

import sys
import types

import torch

# Provide a lightweight dummy vllm module so we can import the hook without the real dependency
if "vllm" not in sys.modules:
    dummy = types.ModuleType("vllm")
    dummy.LLM = object
    dummy.SamplingParams = object
    sys.modules["vllm"] = dummy

from neuro_manipulation.repe.rep_control_vllm_hook import (  # noqa: E402
    _collect_captures_on_worker_rpc,
    _reset_capture_store_for_tests,
    hook_fn_rep_control_with_capture,
)


def setup_function():
    _reset_capture_store_for_tests()


def teardown_function():
    _reset_capture_store_for_tests()


def test_capture_hook_records_pre_and_post_last_token():
    module = types.SimpleNamespace()
    # 3D tensor: (batch, seq, hidden)
    output = torch.zeros(1, 2, 4, dtype=torch.float32)
    module._rep_control_state = {
        "controller": torch.ones(1, 4, dtype=torch.float32),
        "mask": None,
        "token_pos": None,
        "normalize": False,
        "operator_name": "linear_comb",
        "tp_size": 1,
        "capture_id": "run1",
        "layer_id": 3,
    }

    result = hook_fn_rep_control_with_capture(module, (), output)

    # Steering should add the controller everywhere
    assert torch.allclose(result, torch.ones_like(output))

    # Capture store should contain pre/post for the last token position
    captures = _collect_captures_on_worker_rpc(types.SimpleNamespace(rank=0), capture_id="run1")
    assert len(captures) == 1
    entry = captures[0]
    assert entry["capture_id"] == "run1"
    assert entry["layer_id"] == 3
    torch.testing.assert_close(entry["pre"], torch.zeros(1, 4))
    torch.testing.assert_close(entry["post"], torch.ones(1, 4))


def test_collect_clears_store():
    module = types.SimpleNamespace()
    output = torch.zeros(1, 1, 2, dtype=torch.float32)
    module._rep_control_state = {
        "controller": torch.ones(1, 2, dtype=torch.float32),
        "mask": None,
        "token_pos": None,
        "normalize": False,
        "operator_name": "linear_comb",
        "tp_size": 1,
        "capture_id": "run2",
        "layer_id": 1,
    }

    hook_fn_rep_control_with_capture(module, (), output)
    first = _collect_captures_on_worker_rpc(types.SimpleNamespace(rank=0))
    assert first, "expected captures to be returned"

    second = _collect_captures_on_worker_rpc(types.SimpleNamespace(rank=0))
    assert second == [], "capture store should be cleared after collection"
