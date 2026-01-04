"""
Score whether option-probability argmax matches chosen option id for a game_theory_decision run.

This module is intentionally small:
- it builds the exact option strings from `metadata.item_metadata.options`
- it scores logprobs for each option continuation given the stored prompt
- it checks if argmax matches the recorded chosen option id (`score`)

The CLI wiring (loading run_dir, writing CSV) can live here too, but core logic stays testable.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple


def _maybe_tqdm(iterable: Iterable[object], *, total: Optional[int], desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _as_float(v: object) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(v)
    raise TypeError(f"Expected float-compatible value, got {type(v).__name__}")


def _as_int(v: object) -> int:
    if isinstance(v, bool):
        raise TypeError("bool is not a valid int input here")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        return int(float(v))
    raise TypeError(f"Expected int-compatible value, got {type(v).__name__}")


def build_option_strings(options: Sequence[Mapping[str, object]]) -> Dict[int, str]:
    option_strings: Dict[int, str] = {}
    for opt in options:
        opt_id = _as_int(opt["id"])
        text = str(opt["text"])
        option_strings[opt_id] = f"Option {opt_id}. {text}"
    return option_strings


def pick_argmax_option_id(logprobs_by_id: Mapping[int, float]) -> int:
    # Deterministic: break ties by lowest option id.
    return min(logprobs_by_id.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def softmax_from_logprobs(logprobs_by_id: Mapping[int, float]) -> Dict[int, float]:
    if not logprobs_by_id:
        return {}
    m = max(logprobs_by_id.values())
    exps = {k: exp(v - m) for k, v in logprobs_by_id.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


class OptionScorer(Protocol):
    def score_options(self, prompt: str, option_strings: Dict[int, str]) -> Dict[int, float]:
        """Return logprob per option id for each option string continuation."""

    # Optional fast path: implementations may provide this for throughput.
    def score_options_batch(
        self, prompts: Sequence[str], option_strings_list: Sequence[Dict[int, str]]
    ) -> List[Dict[int, float]]:
        """Return logprob dict per prompt in the same order."""


def score_record_match(
    *,
    prompt: str,
    chosen_option_id: int,
    options: Sequence[Mapping[str, object]],
    scorer: OptionScorer,
) -> Dict[str, object]:
    option_strings = build_option_strings(options)
    logprobs_by_id = scorer.score_options(prompt, option_strings)
    predicted_option_id = pick_argmax_option_id(logprobs_by_id)
    probs_by_id = softmax_from_logprobs(logprobs_by_id)
    return {
        "chosen_option_id": int(chosen_option_id),
        "predicted_option_id": int(predicted_option_id),
        "is_match": bool(int(predicted_option_id) == int(chosen_option_id)),
        "prob_by_id": probs_by_id,
        "logprob_by_id": dict(logprobs_by_id),
    }


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_item_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iter_run_records(raw: object) -> Iterable[Mapping[str, object]]:
    if not isinstance(raw, list):
        raise TypeError("raw_results.json must be a list of records")
    for r in raw:
        if isinstance(r, dict):
            yield r


def _group_adjacent_by_emotion_intensity(records: Sequence[Mapping[str, object]]) -> List[List[Mapping[str, object]]]:
    groups: List[List[Mapping[str, object]]] = []
    cur: List[Mapping[str, object]] = []
    cur_key: Optional[Tuple[str, float]] = None
    for r in records:
        key = (str(r["emotion"]), _as_float(r["intensity"]))
        if cur_key is None or key == cur_key:
            cur.append(r)
            cur_key = key
            continue
        groups.append(cur)
        cur = [r]
        cur_key = key
    if cur:
        groups.append(cur)
    return groups


def _options_by_key(raw_records: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, float, int], List[Mapping[str, object]]]:
    out: Dict[Tuple[str, float, int], List[Mapping[str, object]]] = {}
    for r in raw_records:
        if not isinstance(r, dict):
            continue
        emotion = r.get("emotion")
        intensity = r.get("intensity")
        item_id = r.get("item_id")
        if emotion is None or intensity is None or item_id is None:
            continue
        md = r.get("metadata") or {}
        options = (md.get("item_metadata") or {}).get("options")
        if not isinstance(options, list):
            continue
        out[(str(emotion), _as_float(intensity), _as_int(item_id))] = [o for o in options if isinstance(o, dict)]
    return out


def _option_payload_by_id(options: Sequence[Mapping[str, object]]) -> Dict[int, Tuple[str, str]]:
    out: Dict[int, Tuple[str, str]] = {}
    for opt in options:
        if "id" not in opt or "text" not in opt:
            continue
        opt_id = _as_int(opt["id"])
        text = str(opt["text"])
        if "behavior" not in opt:
            raise KeyError(f"Missing behavior for option_id={opt_id}")
        behavior = str(opt["behavior"])
        out[opt_id] = (text, behavior)
    return out


def behavior_match_rows(
    *,
    raw_records: Sequence[Mapping[str, object]],
    scored_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    opts_by_key = _options_by_key(raw_records)

    out: List[Dict[str, object]] = []
    for row in scored_rows:
        emotion = str(row["emotion"])
        intensity = _as_float(row["intensity"])
        item_id = _as_int(row["item_id"])
        pred_id = _as_int(row["predicted_option_id"])
        chosen_id = row.get("chosen_option_id")
        chosen_opt_id = None if chosen_id is None else _as_int(chosen_id)

        options = opts_by_key.get((emotion, intensity, item_id))
        if options is None:
            raise KeyError(f"Missing options for emotion={emotion} intensity={intensity} item_id={item_id}")
        payload_by_id = _option_payload_by_id(options)

        pred_text, pred_behavior = payload_by_id[pred_id]
        if chosen_opt_id is None:
            chosen_text, chosen_behavior, is_match = "", "", None
        else:
            chosen_payload = payload_by_id.get(chosen_opt_id)
            if chosen_payload is None:
                chosen_text, chosen_behavior, is_match = "", "", None
            else:
                chosen_text, chosen_behavior = chosen_payload
                is_match = bool(pred_behavior == chosen_behavior)

        out.append(
            {
                "item_id": item_id,
                "emotion": emotion,
                "intensity": intensity,
                "chosen_option_id": chosen_opt_id,
                "predicted_option_id": pred_id,
                "predicted_option_text": pred_text,
                "predicted_behavior": pred_behavior,
                "chosen_option_text": chosen_text,
                "chosen_behavior": chosen_behavior,
                "is_behavior_match": is_match,
            }
        )
    return out


def enrich_scored_rows_with_behaviors(
    *,
    raw_records: Sequence[Mapping[str, object]],
    scored_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    behavior_rows = behavior_match_rows(raw_records=raw_records, scored_rows=scored_rows)
    by_key: Dict[Tuple[str, float, int], Tuple[str, str]] = {}
    for r in behavior_rows:
        by_key[(str(r["emotion"]), _as_float(r["intensity"]), _as_int(r["item_id"]))] = (
            str(r["chosen_behavior"]),
            str(r["predicted_behavior"]),
        )

    out: List[Dict[str, object]] = []
    for r in scored_rows:
        key = (str(r["emotion"]), _as_float(r["intensity"]), _as_int(r["item_id"]))
        chosen_behavior, predicted_behavior = by_key[key]
        rr = dict(r)
        rr["chosen_behavior"] = chosen_behavior
        rr["predicted_behavior"] = predicted_behavior
        out.append(rr)
    return out


def predicted_option_argmax_ratios(scored_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    counts: Dict[Tuple[str, float], Dict[int, int]] = {}
    totals: Dict[Tuple[str, float], int] = {}
    for r in scored_rows:
        key = (str(r["emotion"]), _as_float(r["intensity"]))
        opt_id = _as_int(r["predicted_option_id"])
        totals[key] = totals.get(key, 0) + 1
        counts.setdefault(key, {})
        counts[key][opt_id] = counts[key].get(opt_id, 0) + 1

    out: List[Dict[str, object]] = []
    for (emotion, intensity), c in sorted(counts.items()):
        total = totals[(emotion, intensity)] or 1
        for opt_id in sorted(c):
            out.append(
                {
                    "emotion": emotion,
                    "intensity": intensity,
                    "option_id": int(opt_id),
                    "ratio": float(c[opt_id]) / float(total),
                }
            )
    return out


def predicted_behavior_argmax_ratios(behavior_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    counts: Dict[Tuple[str, float], Dict[str, int]] = {}
    totals: Dict[Tuple[str, float], int] = {}
    for r in behavior_rows:
        key = (str(r["emotion"]), _as_float(r["intensity"]))
        label = str(r["predicted_behavior"])
        totals[key] = totals.get(key, 0) + 1
        counts.setdefault(key, {})
        counts[key][label] = counts[key].get(label, 0) + 1

    out: List[Dict[str, object]] = []
    for (emotion, intensity), c in sorted(counts.items()):
        total = totals[(emotion, intensity)] or 1
        for label in sorted(c):
            out.append(
                {
                    "emotion": emotion,
                    "intensity": intensity,
                    "behavior_label": label,
                    "ratio": float(c[label]) / float(total),
                }
            )
    return out


def score_records(
    *,
    raw_records: Sequence[Mapping[str, object]],
    scorer: object,
    limit: Optional[int],
    batch_size: int,
    progress: bool,
) -> List[Dict[str, object]]:
    limited: List[Mapping[str, object]] = []
    for idx, r in enumerate(raw_records):
        if limit is not None and idx >= int(limit):
            break
        limited.append(r)

    rows: List[Dict[str, object]] = []
    groups = _group_adjacent_by_emotion_intensity(limited)
    for group in _maybe_tqdm(groups, total=len(groups), desc="Emotion groups", enabled=progress):
        emotion = str(group[0]["emotion"])
        intensity = float(group[0]["intensity"])
        if hasattr(scorer, "set_emotion"):
            scorer.set_emotion(emotion, intensity)  # type: ignore[attr-defined]

        chunk_starts = range(0, len(group), int(batch_size))
        for start in _maybe_tqdm(
            chunk_starts, total=(len(group) + int(batch_size) - 1) // int(batch_size), desc=f"{emotion}:{intensity}", enabled=progress
        ):
            chunk = group[start : start + int(batch_size)]
            prompts = [str(r["prompt"]) for r in chunk]
            item_ids = [int(r["item_id"]) for r in chunk]
            chosen_ids: List[Optional[int]] = []
            for r in chunk:
                chosen_raw = r.get("score")
                chosen_ids.append(int(chosen_raw) if chosen_raw is not None else None)
            options_list: List[List[Mapping[str, object]]] = []
            for r in chunk:
                options = r.get("metadata", {}).get("item_metadata", {}).get("options")
                if not isinstance(options, list):
                    raise TypeError(f"Record item_id={r.get('item_id')} missing metadata.item_metadata.options")
                options_list.append(options)

            option_strings_list = [build_option_strings(opts) for opts in options_list]

            if hasattr(scorer, "score_options_batch"):
                logprobs_list = scorer.score_options_batch(prompts, option_strings_list)  # type: ignore[attr-defined]
            else:
                logprobs_list = [scorer.score_options(p, os) for p, os in zip(prompts, option_strings_list)]  # type: ignore[attr-defined]

            for item_id, chosen_option_id, logprobs_by_id in zip(item_ids, chosen_ids, logprobs_list):
                predicted_option_id = pick_argmax_option_id(logprobs_by_id)
                probs_by_id = softmax_from_logprobs(logprobs_by_id)
                rows.append(
                    {
                        "item_id": int(item_id),
                        "emotion": emotion,
                        "intensity": intensity,
                        "chosen_option_id": chosen_option_id,
                        "predicted_option_id": int(predicted_option_id),
                        "is_match": None if chosen_option_id is None else bool(int(predicted_option_id) == int(chosen_option_id)),
                        "p_option_1": probs_by_id.get(1),
                        "p_option_2": probs_by_id.get(2),
                    }
                )

    return rows


@dataclass(frozen=True)
class VLLMRepEConfig:
    model_path: str
    repe_eng_config: Mapping[str, object]


class VLLMRepEOptionScorer:
    """
    Production scorer: vLLM logprobs + RepE steering per (emotion,intensity).

    Heavy imports are delayed so unit tests can import this module without vLLM.
    """

    def __init__(self, cfg: VLLMRepEConfig):
        self._cfg = cfg
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        import os
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # Ensure custom ReP pipelines ("rep-reading", "rep-control", ...) are registered
        # before `neuro_manipulation.model_utils.load_emotion_readers()` calls HF `pipeline()`.
        from neuro_manipulation.repe.pipelines import repe_pipeline_registry
        repe_pipeline_registry()

        from neuro_manipulation.model_layer_detector import ModelLayerDetector
        from neuro_manipulation.model_utils import load_emotion_readers, setup_model_and_tokenizer
        from neuro_manipulation.repe.sequence_prob_vllm_hook import CombinedVLLMHook
        import torch
        from vllm import LLM

        repe_eng_config = dict(self._cfg.repe_eng_config)
        repe_eng_config["model_name_or_path"] = self._cfg.model_path

        class _VLLMLoadingConfig:
            def __init__(self, model_path: str):
                self.model_path = model_path

            def to_vllm_kwargs(self):
                return {
                    "model": self.model_path,
                    "tensor_parallel_size": 1,
                    "max_model_len": 8192,
                    "trust_remote_code": True,
                    "enforce_eager": True,
                    "gpu_memory_utilization": 0.85,
                    "dtype": "float16",
                    "seed": 42,
                    "disable_custom_all_reduce": False,
                    # The repo already has tests that allow forcing this via env;
                    # setting it here avoids silent backend mismatches on some nodes.
                    "attention_backend": "TRITON_ATTN",
                }

        temp_model, temp_tokenizer, _, _ = setup_model_and_tokenizer(repe_eng_config, from_vllm=False)
        num_layers = ModelLayerDetector.num_layers(temp_model)
        hidden_layers = list(range(-1, -num_layers - 1, -1))
        control_layers = hidden_layers[len(hidden_layers) // 3 : 2 * len(hidden_layers) // 3]
        emotion_readers = load_emotion_readers(
            repe_eng_config,
            temp_model,
            temp_tokenizer,
            hidden_layers,
            enable_thinking=False,
        )
        del temp_model
        del temp_tokenizer
        torch.cuda.empty_cache()

        model, tokenizer, _, _ = setup_model_and_tokenizer(
            _VLLMLoadingConfig(self._cfg.model_path),
            from_vllm=True,
        )
        if not isinstance(model, LLM):
            raise ValueError("Expected vLLM model for logprob scoring")

        self._hook = CombinedVLLMHook(
            model=model,
            tokenizer=tokenizer,
            layers=control_layers,
            block_name=repe_eng_config.get("block_name", "decoder_block"),
            tensor_parallel_size=repe_eng_config.get("tensor_parallel_size", 1),
            enable_sequence_prob=True,
            enable_rep_control=True,
            enable_layer_logit_recording=False,
        )
        self._tokenizer = tokenizer
        self._emotion_readers = emotion_readers
        self._control_layers = control_layers
        self._current_activations = None
        self._initialized = True

    def set_emotion(self, emotion: str, intensity: float) -> None:
        self._lazy_init()
        if emotion == "neutral" or float(intensity) == 0.0:
            self._current_activations = None
            return
        rep_reader = self._emotion_readers.get(emotion)
        if rep_reader is None:
            raise KeyError(f"No RepReader loaded for emotion={emotion}")
        import torch

        self._current_activations = {
            layer: torch.tensor(float(intensity) * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).cpu().half()
            for layer in self._control_layers
        }

    def score_options(self, prompt: str, option_strings: Dict[int, str]) -> Dict[int, float]:
        self._lazy_init()
        # For the game_theory_decision benchmark, the model is instructed to output:
        #   {"decision": "<copy one option text exactly>"}
        # So we score those exact JSON completions, keyed by option id.
        option_texts_by_id = {opt_id: s.split(". ", 1)[1] for opt_id, s in option_strings.items()}
        target_by_id = {opt_id: json.dumps({"decision": option_text}, ensure_ascii=False) for opt_id, option_text in option_texts_by_id.items()}
        target_sequences = [target_by_id[k] for k in sorted(target_by_id)]

        if self._current_activations is not None:
            self._hook._set_control_activations(self._current_activations)
        try:
            prob_results = self._hook.get_log_prob(text_inputs=[prompt], target_sequences=target_sequences)
        finally:
            if self._current_activations is not None:
                self._hook._clear_control_activations()

        logprobs_by_id: Dict[int, float] = {}
        by_seq = {r["sequence"]: r for r in prob_results}
        for opt_id, seq in target_by_id.items():
            row = by_seq.get(seq)
            if row is None:
                raise KeyError(f"Missing logprob result for option_id={opt_id}")
            logprobs_by_id[int(opt_id)] = float(row["log_prob"])
        return logprobs_by_id


class RepControlVLLMOptionScorer:
    """vLLM backend scorer that uses RepControlVLLMHook's string-based worker RPCs (serialization-safe)."""

    def __init__(
        self,
        cfg: VLLMRepEConfig,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
    ):
        self._cfg = cfg
        self._tp_size = int(tensor_parallel_size)
        self._max_model_len = int(max_model_len)
        self._gpu_mem_util = float(gpu_memory_utilization)
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        from neuro_manipulation.repe.pipelines import repe_pipeline_registry

        repe_pipeline_registry()

        from neuro_manipulation.model_layer_detector import ModelLayerDetector
        from neuro_manipulation.model_utils import load_emotion_readers, setup_model_and_tokenizer
        from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook
        import torch
        from vllm import LLM, SamplingParams

        repe_eng_config = dict(self._cfg.repe_eng_config)
        repe_eng_config["model_name_or_path"] = self._cfg.model_path

        class _VLLMLoadingConfig:
            def __init__(
                self,
                model_path: str,
                *,
                tensor_parallel_size: int,
                max_model_len: int,
                gpu_memory_utilization: float,
            ):
                self.model_path = model_path
                self.tensor_parallel_size = int(tensor_parallel_size)
                self.max_model_len = int(max_model_len)
                self.gpu_memory_utilization = float(gpu_memory_utilization)

            def to_vllm_kwargs(self):
                return {
                    "model": self.model_path,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "max_model_len": self.max_model_len,
                    "trust_remote_code": True,
                    "enforce_eager": True,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "dtype": "float16",
                    "seed": 42,
                    "disable_custom_all_reduce": False,
                    "attention_backend": "TRITON_ATTN",
                    "worker_extension_cls": "neuro_manipulation.repe.vllm_worker_extension.NMRepControlWorkerExtension",
                }

        # HF model for RepReaders
        temp_model, temp_tokenizer, _, processor = setup_model_and_tokenizer(repe_eng_config, from_vllm=False)
        num_layers = ModelLayerDetector.num_layers(temp_model)
        hidden_layers = list(range(-1, -num_layers - 1, -1))
        control_layers = hidden_layers[len(hidden_layers) // 3 : 2 * len(hidden_layers) // 3]
        emotion_readers = load_emotion_readers(
            repe_eng_config,
            temp_model,
            temp_tokenizer,
            hidden_layers,
            processor=processor,
            enable_thinking=False,
        )
        del temp_model
        del temp_tokenizer
        torch.cuda.empty_cache()

        # vLLM model for prompt logprobs
        model, tokenizer, _, _ = setup_model_and_tokenizer(
            _VLLMLoadingConfig(
                self._cfg.model_path,
                tensor_parallel_size=self._tp_size,
                max_model_len=self._max_model_len,
                gpu_memory_utilization=self._gpu_mem_util,
            ),
            from_vllm=True,
        )
        if not isinstance(model, LLM):
            raise ValueError("Expected vLLM model for logprob scoring")

        rep_control = RepControlVLLMHook(
            model=model,
            tokenizer=tokenizer,
            layers=control_layers,
            block_name=repe_eng_config.get("block_name", "decoder_block"),
            control_method=repe_eng_config.get("control_method", "reading_vec"),
            tensor_parallel_size=self._tp_size,
        )

        self._model = model
        self._tokenizer = tokenizer
        self._rep_control = rep_control
        self._emotion_readers = emotion_readers
        self._control_layers = control_layers
        self._current_activations = None
        self._initialized = True

    def set_emotion(self, emotion: str, intensity: float) -> None:
        self._lazy_init()
        if emotion == "neutral" or float(intensity) == 0.0:
            self._current_activations = None
            return
        rep_reader = self._emotion_readers.get(emotion)
        if rep_reader is None:
            raise KeyError(f"No RepReader loaded for emotion={emotion}")
        import torch

        self._current_activations = {
            layer: torch.tensor(float(intensity) * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).cpu().half()
            for layer in self._control_layers
        }

    def _set_state(self) -> None:
        if self._current_activations is None:
            return
        # Use the hook's RPC-safe call path by invoking it with max_new_tokens=0 (no generation needed).
        # We'll still explicitly set/reset via RPC to scope the state around our generate() call.
        from neuro_manipulation.repe.rep_control_vllm_hook import _to_rpc_serializable_tensor_payload

        for layer_id, activation_tensor in self._current_activations.items():
            state = {
                "controller": _to_rpc_serializable_tensor_payload(activation_tensor),
                "mask": None,
                "token_pos": None,
                "normalize": False,
                "operator_name": "linear_comb",
                "kwargs": {},
                "tp_size": self._rep_control.tp_size,
            }
            _ = self._model.llm_engine.collective_rpc("_nm_repcontrol_set_state", args=(layer_id, self._rep_control.block_name, state))

    def _reset_state(self) -> None:
        if self._current_activations is None:
            return
        for layer_id in self._current_activations.keys():
            _ = self._model.llm_engine.collective_rpc("_nm_repcontrol_reset_state", args=(layer_id, self._rep_control.block_name))

    def _logprobs_prompt_plus_continuations(self, prompt: str, continuations: Dict[int, str]) -> Dict[int, float]:
        from vllm import SamplingParams

        split_idx = len(self._tokenizer.encode(prompt, add_special_tokens=False))
        full_texts = {opt_id: prompt + cont for opt_id, cont in continuations.items()}

        sampling_params = SamplingParams(max_tokens=1, logprobs=1, prompt_logprobs=1, temperature=0.0)
        outputs = self._model.generate([full_texts[k] for k in sorted(full_texts)], sampling_params, use_tqdm=False)

        logprobs_by_id: Dict[int, float] = {}
        for opt_id, out in zip(sorted(full_texts), outputs):
            prompt_logprobs_list = out.prompt_logprobs
            token_ids = out.prompt_token_ids
            if prompt_logprobs_list is None:
                logprobs_by_id[int(opt_id)] = float("-inf")
                continue

            total = 0.0
            ok = True
            for j, token_id in enumerate(token_ids[split_idx:], start=split_idx):
                lp = prompt_logprobs_list[j]
                if lp is None or token_id not in lp:
                    ok = False
                    break
                total += lp[token_id].logprob
            logprobs_by_id[int(opt_id)] = float(total) if ok else float("-inf")

        return logprobs_by_id

    def score_options_batch(
        self, prompts: Sequence[str], option_strings_list: Sequence[Dict[int, str]]
    ) -> List[Dict[int, float]]:
        self._lazy_init()
        if len(prompts) != len(option_strings_list):
            raise ValueError("prompts and option_strings_list must have same length")

        from vllm import SamplingParams

        # Build (prompt + continuation) for each (item, option), then score continuation tokens via prompt_logprobs.
        full_texts: List[str] = []
        meta: List[Tuple[int, int, int]] = []
        for i, (prompt, option_strings) in enumerate(zip(prompts, option_strings_list)):
            split_idx = len(self._tokenizer.encode(prompt, add_special_tokens=False))
            option_texts_by_id = {opt_id: s.split(". ", 1)[1] for opt_id, s in option_strings.items()}
            continuations = {
                int(opt_id): json.dumps({"decision": option_text}, ensure_ascii=False)
                for opt_id, option_text in option_texts_by_id.items()
            }
            for opt_id in sorted(continuations):
                full_texts.append(prompt + continuations[opt_id])
                meta.append((i, int(opt_id), int(split_idx)))

        sampling_params = SamplingParams(max_tokens=1, logprobs=1, prompt_logprobs=1, temperature=0.0)

        self._set_state()
        try:
            outputs = self._model.generate(full_texts, sampling_params, use_tqdm=False)
        finally:
            self._reset_state()

        out: List[Dict[int, float]] = [dict() for _ in prompts]
        for (prompt_idx, opt_id, split_idx), gen_out in zip(meta, outputs):
            prompt_logprobs_list = gen_out.prompt_logprobs
            token_ids = gen_out.prompt_token_ids
            if prompt_logprobs_list is None:
                out[prompt_idx][opt_id] = float("-inf")
                continue
            total = 0.0
            ok = True
            for j, token_id in enumerate(token_ids[split_idx:], start=split_idx):
                lp = prompt_logprobs_list[j]
                if lp is None or token_id not in lp:
                    ok = False
                    break
                total += lp[token_id].logprob
            out[prompt_idx][opt_id] = float(total) if ok else float("-inf")

        return out

    def score_options(self, prompt: str, option_strings: Dict[int, str]) -> Dict[int, float]:
        self._lazy_init()
        option_texts_by_id = {opt_id: s.split(". ", 1)[1] for opt_id, s in option_strings.items()}
        target_by_id = {opt_id: json.dumps({"decision": option_text}, ensure_ascii=False) for opt_id, option_text in option_texts_by_id.items()}

        self._set_state()
        try:
            return self._logprobs_prompt_plus_continuations(prompt, {int(k): v for k, v in target_by_id.items()})
        finally:
            self._reset_state()


def score_run_dir(
    *,
    run_dir: Path,
    output_csv: Optional[Path] = None,
    limit: Optional[int] = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.85,
    batch_size: int = 256,
    progress: bool = True,
) -> Path:
    raw_path = run_dir / "raw_results.json"
    cfg_path = run_dir / "experiment_config.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    cfg = _load_json(cfg_path)
    if not isinstance(cfg, dict):
        raise TypeError("experiment_config.json must be an object")
    model_path = str(cfg["model_path"])
    repe_eng_config = cfg.get("repe_eng_config")
    if not isinstance(repe_eng_config, dict):
        raise TypeError("experiment_config.json.repe_eng_config must be an object")

    scorer = RepControlVLLMOptionScorer(
        VLLMRepEConfig(model_path=model_path, repe_eng_config=repe_eng_config),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    raw = _load_json(raw_path)
    raw_records_list = list(_iter_run_records(raw))
    rows = score_records(
        raw_records=raw_records_list,
        scorer=scorer,
        limit=limit,
        batch_size=batch_size,
        progress=progress,
    )

    enriched_rows = enrich_scored_rows_with_behaviors(raw_records=raw_records_list, scored_rows=rows)
    out_path = output_csv or (run_dir / "prob_argmax_matches_score.csv")
    _write_item_csv(enriched_rows, out_path)

    behavior_rows = behavior_match_rows(raw_records=raw_records_list, scored_rows=rows)
    _write_item_csv(behavior_rows, run_dir / "behavior_prob_argmax_matches_score.csv")

    _write_item_csv(predicted_option_argmax_ratios(rows), run_dir / "summary_predicted_option_argmax_ratio.csv")
    _write_item_csv(predicted_behavior_argmax_ratios(behavior_rows), run_dir / "summary_predicted_behavior_argmax_ratio.csv")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score argmax(prob) vs chosen option id for a shuffle-choice run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args(argv)
    score_run_dir(
        run_dir=args.run_dir,
        output_csv=args.out_csv,
        limit=args.limit,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
