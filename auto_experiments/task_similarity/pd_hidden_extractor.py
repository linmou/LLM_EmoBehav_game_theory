"""
Responsible: auto_experiments/task-similarity/pd_hidden_extractor.py
Purpose: Extract mean-pooled hidden states over the assistant answer span
         for Prisoner's Dilemma prompts.

This is PD-specific representation code that stays inside auto_experiments.
We do not modify neuro_manipulation or RepReadingPipeline.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def _find_assistant_start(prompt: str) -> int:
    """
    Find the character index where the assistant answer span starts.

    We intentionally use rfind to avoid accidental earlier occurrences of
    'Assistant:' in the description (should not happen in PD prompts, but
    we assert to be safe).
    """
    idx = prompt.rfind("Assistant:")
    if idx == -1:
        raise AssertionError("Prompt missing 'Assistant:' marker")
    first = prompt.find("Assistant:")
    assert first == idx, "Multiple 'Assistant:' markers found in prompt"
    return idx


def collect_answer_means(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    max_length: int,
    batch_size: int = 8,
    span: str = "assistant",
) -> Dict[int, np.ndarray]:
    """
    Collect mean-pooled hidden states over the assistant answer span for each prompt.

    Args:
        model: HF CausalLM with output_hidden_states=True support.
        tokenizer: HF tokenizer.
        prompts: List of full PD prompts (exact strings passed to the model).
        layers: List of transformer layer indices (0-based) to extract from.
                These correspond to model hidden_states[layer+1].
        max_length: max_length used for tokenization/truncation.
        batch_size: how many prompts to process per model forward.
        span: currently only 'assistant' is supported. Other values raise.

    Returns:
        Dict[layer_idx, np.ndarray] with shape (num_prompts, hidden_dim)
        for each requested layer.
    """
    if span != "assistant":
        raise ValueError(f"Unsupported span mode: {span}")

    if not prompts:
        return {int(layer): np.zeros((0, 0), dtype=np.float32) for layer in layers}

    # Compute prefix lengths (in tokens) up to the assistant span start
    prefix_lens: List[int] = []
    for prompt in prompts:
        start_char = _find_assistant_start(prompt)
        prefix_text = prompt[:start_char]
        enc_prefix = tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        # Shape (1, prefix_len)
        prefix_ids = enc_prefix.input_ids[0]
        prefix_len = int(prefix_ids.size(0))
        assert prefix_len > 0, "prefix_len should be > 0"
        prefix_lens.append(prefix_len)

    device = next(model.parameters()).device
    model.eval()

    out: Dict[int, List[np.ndarray]] = {int(l): [] for l in layers}

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        # enc is a simple object with .input_ids and .attention_mask
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states_all = outputs.hidden_states  # tuple len = num_layers + 1

        batch_size_actual = attention_mask.size(0)

        for i in range(batch_size_actual):
            prompt_index = start + i
            seq_len = int(attention_mask[i].sum().item())
            assert seq_len > 0, "sequence length must be > 0"

            prefix_len = prefix_lens[prompt_index]
            # If prefix_len >= seq_len, the assistant answer has been truncated away.
            # This indicates max_length is too small for this prompt.
            assert (
                prefix_len < seq_len
            ), f"Assistant answer truncated (prefix_len={prefix_len}, seq_len={seq_len}). Increase max_length."

            answer_start_tok = prefix_len
            # Sanity check on indices
            assert 0 <= answer_start_tok < seq_len

            for layer in layers:
                layer_idx = int(layer)
                # hidden_states_all: [0]=embedding, [1]=layer0, ...
                hs = hidden_states_all[layer_idx + 1][i, :seq_len, :]  # (seq_len, hidden)
                answer_hs = hs[answer_start_tok:, :]  # (N_answer_tokens_effective, hidden)
                assert (
                    answer_hs.shape[0] > 0
                ), "Answer span produced zero tokens; check prompt formatting and max_length."
                mean_vec = answer_hs.mean(dim=0).detach().cpu().numpy().astype(np.float32)
                out[layer_idx].append(mean_vec)

    return {layer: np.stack(vecs, axis=0) for layer, vecs in out.items()}
