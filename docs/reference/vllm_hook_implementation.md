# vLLM Hook Implementation for Representation Control

This document explains the `RepControlVLLMHook` class, designed to apply representation control techniques (specifically `reading_vec` for now) to models running within the vLLM framework by leveraging forward hooks and Remote Procedure Calls (RPC).

## Overview

Instead of wrapping model layers (as in `rep_control_reading_vec.py`), this approach injects control by registering PyTorch forward hooks directly onto the target layers (or submodules like `mlp`, `self_attn`) of the model running on each vLLM worker process. This avoids modifying the model structure itself but relies on vLLM's `collective_rpc` mechanism to manage the hooks and their associated state.

## How it Works

1.  **Initialization (`__init__`)**:
    *   Takes a vLLM `LLM` instance, tokenizer, target layer indices, the name of the block/module within the layer to hook (e.g., `"decoder_block"` for the layer's main output), and the control method (`"reading_vec"`).
    *   Uses `collective_rpc` to call `_register_hook_on_worker_rpc` on each worker.
    *   `_register_hook_on_worker_rpc` finds the specified module (e.g., the Nth decoder layer) on the worker's copy of the model and registers the `hook_fn_rep_control` function as a forward hook.
    *   The hook initially does nothing, as its control state is not yet set.

2.  **Generation (`__call__`)**:
    *   Takes prompts and optional control parameters (`activations`, `token_pos`, `masks`, `normalize`, `operator`).
    *   **Set State**: If `activations` (a dictionary mapping layer indices to control tensors) are provided, it calls `_set_controller_state_on_worker_rpc` via `collective_rpc`.
        *   This RPC function finds the target module on the worker and attaches a `_rep_control_state` attribute to it. This attribute holds the control tensor, mask, operator function, and other parameters needed by the hook.
    *   **Run Inference**: It calls the standard `model.generate()` method.
        *   During the forward pass on each worker, when execution reaches a hooked module, the `hook_fn_rep_control` is triggered.
        *   The hook checks if `_rep_control_state` exists on the module.
        *   If the state exists, the hook retrieves the control parameters (controller tensor, mask, operator, etc.) and applies the modification logic (e.g., adding the controller vector to the module's output) before returning the modified output.
        *   If no state exists, the hook simply returns the original output.
    *   **Reset State**: After generation finishes (in a `finally` block to ensure cleanup), it calls `_reset_controller_state_on_worker_rpc` via `collective_rpc`.
        *   This RPC function finds the target module on the worker and deletes the `_rep_control_state` attribute, ensuring subsequent unrelated inference calls are not affected.

3.  **Hook Function (`hook_fn_rep_control`)**:
    *   This function contains the core logic for applying the representation control modification, similar to the logic within `WrappedBlock.forward` in `rep_control_reading_vec.py`.
    *   It handles accessing the correct output tensor (even if the module returns a tuple), applying masks, handling token positions, performing normalization, and using the specified operator (e.g., linear combination).

## Advantages

*   **No Model Monkey-Patching**: Doesn't require modifying the vLLM model's layer structure directly.
*   **Leverages vLLM Infrastructure**: Uses `collective_rpc` for distributed state management.

## Disadvantages/Considerations

*   **RPC Overhead**: Sending control state via RPC for every controlled generation call might introduce some overhead compared to having the logic permanently wrapped in the layer.
*   **State Management Complexity**: Relies on correctly setting and resetting state via RPC. Errors in RPC or state management could lead to inconsistent behavior.
*   **Hook Limitations**: Hooks might interact unexpectedly with vLLM's internal optimizations or execution graph. `enforce_eager=True` might be necessary when initializing the vLLM `LLM` object.
*   **Hook Removal**: Properly removing hooks registered via RPC requires careful handle management, which is currently implemented conceptually but might need refinement.

## Usage Example

For an example demonstrating initialization, baseline generation, and controlled generation, please refer to the `if __name__ == "__main__":` block in the `neuro_manipulation/repe/rep_control_vllm_hook.py` script. 