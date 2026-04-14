#!/usr/bin/env python3
# Purpose: analyze repeat-to-repeat sign stability of emotion-vs-neutral behavior deltas for game-theory runs.

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


PROSOCIAL_BEHAVIOR_BY_TASK: Dict[str, str] = {
    "Prisoners_Dilemma": "cooperate",
    "Stag_Hunt": "cooperate",
    "Escalation_Game": "withdraw",
    "Trust_Game_Trustor": "trust_high",
    "Trust_Game_Trustee": "return_high",
    "Ultimatum_Game_Proposer": "offer_high",
    "Ultimatum_Game_Responder": "accept",
}

_RUN_DIR_RE = re.compile(
    r"^(?P<model>.+)_game_theory(_decision)?_(?P<task>.+)_(?P<ts>\d{8}_\d{6})$"
)


@dataclass(frozen=True)
class RunRef:
    model: str
    task: str
    timestamp: str
    dir_path: Path


@dataclass(frozen=True)
class RepeatStabilityOutputs:
    csv_path: Path
    report_path: Path


def _parse_run_dir(name: str) -> Optional[Tuple[str, str, str]]:
    match = _RUN_DIR_RE.match(name)
    if not match:
        return None
    return match.group("model"), match.group("task"), match.group("ts")


def _discover_latest_runs(root: Path) -> List[RunRef]:
    latest: Dict[Tuple[str, str], RunRef] = {}
    for run_dir in root.glob("**/"):
        if not run_dir.is_dir():
            continue
        parsed = _parse_run_dir(run_dir.name)
        if not parsed:
            continue
        model, task, ts = parsed
        ref = RunRef(model=model, task=task, timestamp=ts, dir_path=run_dir)
        key = (model, task)
        if key not in latest or ts > latest[key].timestamp:
            latest[key] = ref
    return [latest[key] for key in sorted(latest)]


def _read_summary_behavior_ratio_by_repeat(path: Path) -> List[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mcnemar_exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n <= 0:
        return 1.0
    k = min(n01, n10)
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i)
    return min(1.0, 2.0 * total / (2.0**n))


def _bootstrap_ci_mean_delta(
    deltas: List[int],
    *,
    samples: int = 400,
    seed: int = 0,
) -> Tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means: List[float] = []
    for _ in range(samples):
        total = 0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return lo, hi


def _bh_fdr(indexed_p_values: List[Tuple[int, float]]) -> Dict[int, float]:
    total = len(indexed_p_values)
    if total == 0:
        return {}
    ranked = sorted(indexed_p_values, key=lambda item: item[1])
    q_raw: List[Tuple[int, float]] = []
    for rank, (idx, p_value) in enumerate(ranked, start=1):
        q_raw.append((idx, min(1.0, p_value * total / rank)))
    q_values: Dict[int, float] = {}
    prev = 1.0
    for idx, q_value in reversed(q_raw):
        prev = min(prev, q_value)
        q_values[idx] = prev
    return q_values


def _extract_choice_and_behavior(record: dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None, None
    item_metadata = metadata.get("item_metadata")
    if not isinstance(item_metadata, dict):
        return None, None
    options = item_metadata.get("options")
    if not isinstance(options, list):
        return None, None

    option_by_text: Dict[str, dict[str, object]] = {}
    for option in options:
        if not isinstance(option, dict):
            continue
        text = option.get("text")
        if isinstance(text, str):
            option_by_text[" ".join(text.strip().lower().split())] = option

    response = record.get("response")
    decision_text: Optional[str] = None
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            decision_text = response
        else:
            if isinstance(parsed, dict) and isinstance(parsed.get("decision"), str):
                decision_text = parsed["decision"]
    if not decision_text:
        return None, None

    chosen = option_by_text.get(" ".join(decision_text.strip().lower().split()))
    if not chosen:
        return None, None
    behavior = chosen.get("behavior")
    if not isinstance(behavior, str):
        return None, None
    return decision_text, behavior


def _build_significance_map(
    run: RunRef,
    target_behavior: str,
) -> Dict[Tuple[str, float], dict[str, object]]:
    raw_path = run.dir_path / "raw_results.json"
    if not raw_path.exists():
        return {}
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {}

    neutral_by_key: Dict[Tuple[int, int], str] = {}
    emotion_by_key: Dict[Tuple[str, float, int, int], str] = {}
    emotion_keys: set[Tuple[str, float]] = set()

    for record in raw:
        if not isinstance(record, dict):
            continue
        emotion = record.get("emotion")
        if not isinstance(emotion, str) or not emotion:
            continue
        try:
            intensity = float(record.get("intensity", 0.0))  # type: ignore[arg-type]
            item_id = int(record.get("item_id"))  # type: ignore[arg-type]
            repeat_id = int(record.get("repeat_id"))  # type: ignore[arg-type]
        except Exception:
            continue
        _, behavior = _extract_choice_and_behavior(record)
        if behavior is None:
            continue
        if emotion == "neutral":
            neutral_by_key[(item_id, repeat_id)] = behavior
            continue
        emotion_keys.add((emotion, intensity))
        emotion_by_key[(emotion, intensity, item_id, repeat_id)] = behavior

    out: Dict[Tuple[str, float], dict[str, object]] = {}
    for emotion, intensity in sorted(emotion_keys):
        deltas: List[int] = []
        neutral_hits = 0
        emotion_hits = 0
        n01 = 0
        n10 = 0
        n_pairs = 0
        for (item_id, repeat_id), neutral_behavior in neutral_by_key.items():
            emotion_behavior = emotion_by_key.get((emotion, intensity, item_id, repeat_id))
            if emotion_behavior is None:
                continue
            n_pairs += 1
            neutral_hit = 1 if neutral_behavior == target_behavior else 0
            emotion_hit = 1 if emotion_behavior == target_behavior else 0
            neutral_hits += neutral_hit
            emotion_hits += emotion_hit
            deltas.append(emotion_hit - neutral_hit)
            if neutral_hit == 0 and emotion_hit == 1:
                n01 += 1
            elif neutral_hit == 1 and emotion_hit == 0:
                n10 += 1
        if n_pairs == 0:
            continue
        p_value = _mcnemar_exact_p(n01, n10)
        ci_low, ci_high = _bootstrap_ci_mean_delta(deltas)
        out[(emotion, intensity)] = {
            "n_pairs": n_pairs,
            "pooled_delta": (emotion_hits / n_pairs) - (neutral_hits / n_pairs),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "q_value": 1.0,
            "significant": False,
        }
    return out


def _sign_symbol(value: float) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def _format_repeat_deltas(deltas: Iterable[float]) -> str:
    return "[" + ",".join(f"{delta:+.6f}" for delta in deltas) + "]"


def _build_rows_for_run(run: RunRef) -> List[dict[str, object]]:
    csv_path = run.dir_path / "summary_behavior_ratio_by_repeat.csv"
    if not csv_path.exists():
        return []

    rows = _read_summary_behavior_ratio_by_repeat(csv_path)
    target_behavior = PROSOCIAL_BEHAVIOR_BY_TASK.get(run.task)
    if target_behavior is None:
        return []
    significance_by_key = _build_significance_map(run, target_behavior)

    neutral_ratio_by_repeat: Dict[int, List[float]] = {}
    emotion_ratio_by_key: Dict[Tuple[str, float, int], float] = {}
    emotion_repeat_ids: Dict[Tuple[str, float], set[int]] = {}

    for row in rows:
        emotion = str(row.get("emotion", "")).strip()
        behavior = str(row.get("behavior", row.get("behavior_label", ""))).strip()
        if not emotion:
            continue
        try:
            intensity = float(row.get("intensity", "0"))
            repeat_id = int(float(row.get("repeat_id", "0")))
            ratio = float(row.get("ratio", "0"))
        except ValueError:
            continue

        if emotion == "neutral":
            if behavior == target_behavior:
                neutral_ratio_by_repeat.setdefault(repeat_id, []).append(ratio)
            else:
                neutral_ratio_by_repeat.setdefault(repeat_id, [])
            continue

        emotion_repeat_ids.setdefault((emotion, intensity), set()).add(repeat_id)
        if behavior == target_behavior:
            emotion_ratio_by_key[(emotion, intensity, repeat_id)] = ratio

    neutral_mean_by_repeat = {
        repeat_id: (mean(values) if values else 0.0)
        for repeat_id, values in neutral_ratio_by_repeat.items()
    }

    out: List[dict[str, object]] = []
    for (emotion, intensity), repeat_ids in sorted(emotion_repeat_ids.items()):
        common_repeat_ids = sorted(set(neutral_mean_by_repeat) & repeat_ids)
        if not common_repeat_ids:
            continue
        repeat_deltas: List[float] = []
        for repeat_id in common_repeat_ids:
            emotion_ratio = emotion_ratio_by_key.get((emotion, intensity, repeat_id), 0.0)
            neutral_ratio = neutral_mean_by_repeat[repeat_id]
            repeat_deltas.append(emotion_ratio - neutral_ratio)

        sign_pattern = "".join(_sign_symbol(delta) for delta in repeat_deltas)
        nonzero_signs = {symbol for symbol in sign_pattern if symbol != "0"}
        out.append(
            {
                "model": run.model,
                "task": run.task,
                "target_behavior": target_behavior,
                "emotion": emotion,
                "intensity": intensity,
                "repeat_ids": ",".join(str(repeat_id) for repeat_id in common_repeat_ids),
                "repeat_deltas": _format_repeat_deltas(repeat_deltas),
                "repeat_sign_pattern": sign_pattern,
                "flip_across_repeats": len(nonzero_signs) > 1,
                "mean_delta": mean(repeat_deltas),
                "min_delta": min(repeat_deltas),
                "max_delta": max(repeat_deltas),
                "n_pairs": "",
                "pooled_delta": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "q_value": "",
                "significant": "",
                "source_run_dir": str(run.dir_path),
            }
        )
        sig = significance_by_key.get((emotion, intensity))
        if sig is not None:
            out[-1].update(sig)
    return out


def _write_csv(path: Path, rows: List[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "task",
        "target_behavior",
        "emotion",
        "intensity",
        "repeat_ids",
        "repeat_deltas",
        "repeat_sign_pattern",
        "flip_across_repeats",
        "mean_delta",
        "min_delta",
        "max_delta",
        "n_pairs",
        "pooled_delta",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "significant",
        "source_run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: List[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    flip_count = sum(1 for row in rows if bool(row["flip_across_repeats"]))
    sig_count = sum(1 for row in rows if bool(row["significant"]))
    lines = [
        "# Repeat Stability Report",
        "",
        "Intent: summarize whether emotion-vs-neutral behavior deltas keep the same sign across repeats or flip, and attach paired significance when raw results are available.",
        "",
        f"- total_conditions: {total}",
        f"- flip_across_repeats: {flip_count}",
        f"- significant_q_lt_0_05: {sig_count}",
        "",
        "| model | task | emotion | intensity | target_behavior | repeat_sign_pattern | flip_across_repeats | mean_delta | pooled_delta | q_value | significant |",
        "| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        model = str(row["model"])
        task = str(row["task"])
        emotion = str(row["emotion"])
        intensity = float(str(row["intensity"]))
        target_behavior = str(row["target_behavior"])
        repeat_sign_pattern = str(row["repeat_sign_pattern"])
        flip_across_repeats = bool(row["flip_across_repeats"])
        mean_delta = float(str(row["mean_delta"]))
        pooled_delta = str(row["pooled_delta"]) if row["pooled_delta"] != "" else ""
        q_value = str(row["q_value"]) if row["q_value"] != "" else ""
        significant = bool(row["significant"]) if row["significant"] != "" else ""
        pooled_delta_cell = f"{float(pooled_delta):+.6f}" if pooled_delta else ""
        q_value_cell = f"{float(q_value):.6f}" if q_value else ""
        lines.append(
            f"| {model} | {task} | {emotion} | {intensity:.1f} | "
            f"{target_behavior} | {repeat_sign_pattern} | {flip_across_repeats} | {mean_delta:+.6f} | "
            f"{pooled_delta_cell} | {q_value_cell} | {significant} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_repeat_stability(root: Path, out_dir: Optional[Path] = None) -> RepeatStabilityOutputs:
    root = root.resolve()
    out_root = out_dir.resolve() if out_dir is not None else root
    rows: List[dict[str, object]] = []
    for run in _discover_latest_runs(root):
        rows.extend(_build_rows_for_run(run))

    rows.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["task"]),
            str(row["emotion"]),
            float(str(row["intensity"])),
        )
    )
    indexed_p_values: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows):
        p_value = row["p_value"]
        if p_value == "":
            continue
        indexed_p_values.append((idx, float(str(p_value))))
    q_values = _bh_fdr(indexed_p_values)
    for idx, q_value in q_values.items():
        rows[idx]["q_value"] = q_value
        ci_low = float(str(rows[idx]["ci_low"]))
        ci_high = float(str(rows[idx]["ci_high"]))
        rows[idx]["significant"] = q_value < 0.05 and not (ci_low <= 0.0 <= ci_high)

    csv_path = out_root / "repeat_stability_analysis.csv"
    report_path = out_root / "repeat_stability_report.md"
    _write_csv(csv_path, rows)
    _write_report(report_path, rows)
    return RepeatStabilityOutputs(csv_path=csv_path, report_path=report_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze repeat-to-repeat sign stability of emotion-vs-neutral behavior deltas."
    )
    parser.add_argument("root", type=Path, help="Results root that contains game-theory run directories")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory for the CSV and Markdown report",
    )
    args = parser.parse_args()
    outputs = analyze_repeat_stability(root=args.root, out_dir=args.out_dir)
    print(outputs.csv_path)
    print(outputs.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
