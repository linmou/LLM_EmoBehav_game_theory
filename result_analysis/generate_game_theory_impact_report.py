"""Generate option- and (when available) behavior-level emotion impact reports (vs neutral).

Designed to work with both:
- `results/new_game_theory_decision/shuffle_choices/` (has choice + behavior ratios)
- `results/new_game_theory/` (typically has choice ratios only)

Inputs (per run directory, when present):
- summary_choice_ratio.csv: emotion,intensity,option_id,ratio
- summary_behavior_ratio.csv: emotion,intensity,behavior,ratio

Method:
1) For each (model, game_setting), keep the latest timestamped run directory.
2) Ignore intensity by collapsing with mean(ratio) over intensities.
3) Compute per emotion delta vs neutral for each option_id / behavior:
     delta = ratio(emotion, x) - ratio(neutral, x)
4) Summarize per (model, game_setting, x) the best/worst deltas and range.

Outputs (written under root):
- option_impacted_by_emo_vs_neutral_latest.csv
- behavior_impacted_emo_vs_neutral_latest.csv (only if behavior inputs exist)
- option_intensity_impacted_by_emo_vs_neutral_latest.csv
- behavior_intensity_impacted_emo_vs_neutral_latest.csv (only if behavior inputs exist)
- game_theory_impact_report.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from result_analysis.game_theory_impact_heatmaps import write_behavior_change_heatmap_pdf
from result_analysis.game_theory_ratio_loading import (
    collapse_behavior_over_intensity,
    collapse_choice_over_intensity,
    load_behavior_by_intensity,
    load_choice_by_intensity,
)


_EXPECTED_SCORE_SPECS: Dict[str, Dict[str, float]] = {
    "Trust_Game_Trustor": {"trust_none": 0.0, "trust_low": 1.0, "trust_high": 2.0},
    "Trust_Game_Trustee": {"return_none": 0.0, "return_medium": 1.0, "return_high": 2.0},
    "Ultimatum_Game_Proposer": {"offer_low": 0.0, "offer_medium": 1.0, "offer_high": 2.0},
    "Ultimatum_Game_Responder": {"reject": 0.0, "accept": 1.0},
}


_RUN_DIR_RE = re.compile(
    r"^(?P<model>.+)_game_theory(_decision)?_(?P<task>.+)_(?P<ts>\d{8}_\d{6})$"
)

UNKNOWN_MIN_RATIO = 0.01
BOOTSTRAP_SAMPLES = 400


@dataclass(frozen=True)
class RunRef:
    model: str
    task: str
    timestamp: str
    dir_path: Path


@dataclass(frozen=True)
class ReportOutputs:
    option_csv_path: Path
    behavior_csv_path: Optional[Path]
    option_intensity_csv_path: Path
    behavior_intensity_csv_path: Optional[Path]
    heatmaps_dir: Optional[Path]
    report_path: Path


@dataclass(frozen=True)
class SigEntry:
    n_pairs: int
    neutral_rate: float
    emotion_rate: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float

    def stars(self) -> str:
        if self.q_value < 0.001:
            return "!!!"
        if self.q_value < 0.01:
            return "!!"
        if self.q_value < 0.05:
            return "!"
        return ""


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _norm_decision_match_text(s: str) -> str:
    return _norm_text(s).rstrip(" .,!?:;\"'")


def _format_ranked_deltas(deltas: List[Tuple[str, float, float]]) -> str:
    """
    Format (emotion, delta_vs_neutral, ratio) into a stable, readable string.

    Ranked by delta desc, then emotion asc.
    """
    ranked = sorted(deltas, key=lambda x: (-x[1], x[0]))
    return ";".join(f"{emo}:{delta:+.6f}" for emo, delta, _ in ranked)


def _filter_unknown_if_rare(deltas: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    return [d for d in deltas if not (d[0] == "unknown" and d[2] < UNKNOWN_MIN_RATIO)]


def _parse_delta_string(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        emo, val = part.split(":", 1)
        emo = emo.strip()
        try:
            out[emo] = float(val)
        except ValueError:
            continue
    return out


def _parse_run_dir(name: str) -> Optional[Tuple[str, str, str]]:
    match = _RUN_DIR_RE.match(name)
    if not match:
        return None
    return match.group("model"), match.group("task"), match.group("ts")


def _discover_latest_runs(root: Path) -> List[RunRef]:
    latest: Dict[Tuple[str, str], RunRef] = {}
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        parsed = _parse_run_dir(run_dir.name)
        if not parsed:
            continue
        model, task, ts = parsed
        key = (model, task)
        candidate = RunRef(model=model, task=task, timestamp=ts, dir_path=run_dir)
        if key not in latest or ts > latest[key].timestamp:
            latest[key] = candidate
    return [latest[k] for k in sorted(latest)]


def _mcnemar_exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n <= 0:
        return 1.0
    k = min(n01, n10)
    # exact two-sided via binomial(n, 0.5)
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i)
    p = 2.0 * total / (2.0**n)
    return min(1.0, p)


def _bootstrap_ci_mean_delta(deltas: List[int], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> Tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means: List[float] = []
    for _ in range(samples):
        s = 0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return lo, hi


def _load_chosen_behavior_from_detailed_csv(path: Path) -> List[Tuple[str, int, int, str]]:
    """
    Returns list of (emotion, item_id, repeat_id, chosen_behavior).
    """
    out: List[Tuple[str, int, int, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emo = str(row.get("emotion", ""))
            if not emo:
                continue
            try:
                item_id = int(float(row.get("item_id", "0")))
                repeat_id = int(float(row.get("repeat_id", "0")))
            except Exception:
                continue
            beh = row.get("chosen_behavior")
            if not isinstance(beh, str) or not beh:
                continue
            out.append((emo, item_id, repeat_id, beh))
    return out


def _load_raw_results_list(raw_path: Path) -> List[object]:
    text = raw_path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        try:
            raw, end = decoder.raw_decode(text)
        except json.JSONDecodeError:
            print(f"WARNING: skipping invalid JSON: {raw_path} ({exc})", file=sys.stderr)
            return []
        trailing = text[end:].strip()
        if trailing:
            print(
                f"WARNING: recovered first JSON value from {raw_path} and ignored trailing data after char {end}",
                file=sys.stderr,
            )
    if not isinstance(raw, list):
        return []
    return raw


def _item_change_rates_for_run(run: RunRef) -> Dict[str, Tuple[float, int]]:
    """
    Per emotion: fraction of paired items where chosen_behavior != neutral chosen_behavior.

    Uses (item_id, repeat_id) pairing and ignores intensity.
    """
    detailed_path = run.dir_path / "detailed_results.csv"
    rows: List[Tuple[str, int, int, str]] = []
    if detailed_path.exists():
        rows = _load_chosen_behavior_from_detailed_csv(detailed_path)
    else:
        raw_path = run.dir_path / "raw_results.json"
        if not raw_path.exists():
            return {}
        raw = _load_raw_results_list(raw_path)
        if not raw:
            return {}
        for rec in raw:
            if not isinstance(rec, dict):
                continue
            emo = rec.get("emotion")
            if not isinstance(emo, str) or not emo:
                continue
            try:
                item_id = int(rec.get("item_id"))  # type: ignore[arg-type]
                repeat_id = int(rec.get("repeat_id"))  # type: ignore[arg-type]
            except Exception:
                continue
            _, beh = _extract_choice_and_behavior(rec)
            if not isinstance(beh, str) or not beh:
                continue
            rows.append((emo, item_id, repeat_id, beh))

    neutral: Dict[Tuple[int, int], str] = {}
    emo_map: Dict[str, Dict[Tuple[int, int], str]] = {}
    for emo, item_id, repeat_id, beh in rows:
        key = (item_id, repeat_id)
        if emo == "neutral":
            neutral[key] = beh
        else:
            emo_map.setdefault(emo, {})[key] = beh

    if not neutral:
        return {}

    rates: Dict[str, Tuple[float, int]] = {}
    for emo, m in emo_map.items():
        keys = neutral.keys() & m.keys()
        n = len(keys)
        if n == 0:
            continue
        changed = sum(1 for k in keys if m[k] != neutral[k])
        rates[emo] = (changed / n, n)
    return rates


def _bh_fdr(p_values: List[Tuple[Tuple[str, object, str], float]]) -> Dict[Tuple[str, object, str], float]:
    # keys are (kind, row_key, emotion)
    m = len(p_values)
    if m == 0:
        return {}
    ranked = sorted(p_values, key=lambda x: x[1])
    q_raw: List[Tuple[Tuple[str, object, str], float]] = []
    for i, (k, p) in enumerate(ranked, start=1):
        q_raw.append((k, min(1.0, p * m / i)))
    # enforce monotonicity from largest p downwards
    q: Dict[Tuple[str, object, str], float] = {}
    prev = 1.0
    for k, qv in reversed(q_raw):
        prev = min(prev, qv)
        q[k] = prev
    return q


def _extract_choice_and_behavior(record: Dict[str, object]) -> Tuple[Optional[int], Optional[str]]:
    md = record.get("metadata")
    if not isinstance(md, dict):
        return None, None
    item_md = md.get("item_metadata")
    if not isinstance(item_md, dict):
        return None, None
    options = item_md.get("options")
    if not isinstance(options, list):
        return None, None
    opt_by_text: Dict[str, Dict[str, object]] = {}
    for o in options:
        if not isinstance(o, dict):
            continue
        text = o.get("text")
        if not isinstance(text, str):
            continue
        opt_by_text[_norm_decision_match_text(text)] = o

    resp = record.get("response")
    decision_text: Optional[str] = None
    if isinstance(resp, dict) and isinstance(resp.get("decision"), str):
        decision_text = resp["decision"]
    elif isinstance(resp, str):
        try:
            parsed = json.loads(resp)
            if isinstance(parsed, dict) and isinstance(parsed.get("decision"), str):
                decision_text = parsed["decision"]
        except Exception:
            try:
                parsed = ast.literal_eval(resp)
                if isinstance(parsed, dict) and isinstance(parsed.get("decision"), str):
                    decision_text = parsed["decision"]
                else:
                    decision_text = resp
            except Exception:
                decision_text = resp
    if not decision_text:
        return None, None
    chosen = opt_by_text.get(_norm_decision_match_text(decision_text))
    if not chosen:
        return None, None
    opt_id = chosen.get("id")
    behavior = chosen.get("behavior")
    if behavior in (None, ""):
        behavior = chosen.get("behavior_label")
    if not isinstance(opt_id, int) or not isinstance(behavior, str):
        return None, None
    return opt_id, behavior


def _peak_choice_sig_intensities(
    csv_path: Path,
    *,
    unknown_threshold: Optional[float],
) -> Dict[Tuple[int, str], float]:
    by_intensity, _ = load_choice_by_intensity(csv_path, unknown_threshold=unknown_threshold)
    if "neutral" not in by_intensity:
        return {}
    neutral_ints = sorted(by_intensity["neutral"])
    option_ids = sorted({option_id for emo in by_intensity for intensity in by_intensity[emo] for option_id in by_intensity[emo][intensity]})
    for intensity in neutral_ints:
        for option_id in option_ids:
            by_intensity["neutral"][intensity].setdefault(option_id, 0.0)
    neutral_base = {
        option_id: mean([by_intensity["neutral"][intensity][option_id] for intensity in neutral_ints])
        for option_id in option_ids
    }

    chosen: Dict[Tuple[int, str], float] = {}
    for emo in sorted(e for e in by_intensity if e != "neutral"):
        for option_id in option_ids:
            best_delta: Optional[float] = None
            best_intensity: Optional[float] = None
            for intensity, ratios in by_intensity[emo].items():
                delta = float(ratios.get(option_id, 0.0)) - neutral_base[option_id]
                if best_delta is None or abs(delta) > abs(best_delta) or (
                    abs(delta) == abs(best_delta) and best_intensity is not None and intensity < best_intensity
                ):
                    best_delta = delta
                    best_intensity = float(intensity)
            if best_intensity is not None:
                chosen[(option_id, emo)] = best_intensity
    return chosen


def _peak_behavior_sig_intensities(
    csv_path: Path,
    *,
    unknown_threshold: Optional[float],
) -> Dict[Tuple[str, str], float]:
    by_intensity, _ = load_behavior_by_intensity(csv_path, unknown_threshold=unknown_threshold)
    if "neutral" not in by_intensity:
        return {}
    neutral_ints = sorted(by_intensity["neutral"])
    behaviors = sorted({behavior for emo in by_intensity for intensity in by_intensity[emo] for behavior in by_intensity[emo][intensity]})
    for intensity in neutral_ints:
        for behavior in behaviors:
            by_intensity["neutral"][intensity].setdefault(behavior, 0.0)
    neutral_base = {
        behavior: mean([by_intensity["neutral"][intensity][behavior] for intensity in neutral_ints])
        for behavior in behaviors
    }

    chosen: Dict[Tuple[str, str], float] = {}
    for emo in sorted(e for e in by_intensity if e != "neutral"):
        for behavior in behaviors:
            best_delta: Optional[float] = None
            best_intensity: Optional[float] = None
            for intensity, ratios in by_intensity[emo].items():
                delta = float(ratios.get(behavior, 0.0)) - neutral_base[behavior]
                if best_delta is None or abs(delta) > abs(best_delta) or (
                    abs(delta) == abs(best_delta) and best_intensity is not None and intensity < best_intensity
                ):
                    best_delta = delta
                    best_intensity = float(intensity)
            if best_intensity is not None:
                chosen[(behavior, emo)] = best_intensity
    return chosen


def _sig_maps_for_run(
    run: RunRef,
    *,
    unknown_threshold: Optional[float],
) -> Tuple[Dict[int, Dict[str, SigEntry]], Dict[str, Dict[str, SigEntry]]]:
    """
    Compute significance maps (vs neutral) from raw per-item results.

    Returns:
      option_sig[option_id][emotion] = SigEntry
      behavior_sig[behavior][emotion] = SigEntry
    """
    raw_path = run.dir_path / "raw_results.json"
    if not raw_path.exists():
        return {}, {}

    raw = _load_raw_results_list(raw_path)
    if not raw:
        return {}, {}

    choice_csv = run.dir_path / "summary_choice_ratio.csv"
    behavior_csv = run.dir_path / "summary_behavior_ratio.csv"
    peak_choice_intensity = (
        _peak_choice_sig_intensities(choice_csv, unknown_threshold=unknown_threshold) if choice_csv.exists() else {}
    )
    peak_behavior_intensity = (
        _peak_behavior_sig_intensities(behavior_csv, unknown_threshold=unknown_threshold) if behavior_csv.exists() else {}
    )

    # Pair neutral by scenario identity, but evaluate each emotion row at its peak-|delta| intensity.
    # (emotion, intensity, item_id, repeat_id) -> chosen option/behavior
    obs: Dict[Tuple[str, float, int, int], Tuple[int, str]] = {}
    option_ids: set[int] = set()
    behaviors: set[str] = set()
    emotions: set[str] = set()
    neutral_by_key: Dict[Tuple[int, int], Tuple[float, Tuple[int, str]]] = {}
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        emo = rec.get("emotion")
        if not isinstance(emo, str):
            continue
        try:
            intensity = float(rec.get("intensity", 0.0))  # type: ignore[arg-type]
        except Exception:
            intensity = 0.0
        emotions.add(emo)
        try:
            item_id = int(rec.get("item_id"))  # type: ignore[arg-type]
            repeat_id = int(rec.get("repeat_id"))  # type: ignore[arg-type]
        except Exception:
            continue
        opt_id, beh = _extract_choice_and_behavior(rec)
        if opt_id is None or beh is None:
            continue
        option_ids.add(opt_id)
        behaviors.add(beh)
        obs[(emo, intensity, item_id, repeat_id)] = (opt_id, beh)
        if emo == "neutral":
            key = (item_id, repeat_id)
            current = neutral_by_key.get(key)
            if current is None or intensity < current[0]:
                neutral_by_key[key] = (intensity, (opt_id, beh))

    if "neutral" not in emotions:
        return {}, {}

    option_sig: Dict[int, Dict[str, SigEntry]] = {oid: {} for oid in sorted(option_ids)}
    behavior_sig: Dict[str, Dict[str, SigEntry]] = {b: {} for b in sorted(behaviors)}

    option_pvals: List[Tuple[Tuple[str, object, str], float]] = []
    behavior_pvals: List[Tuple[Tuple[str, object, str], float]] = []

    for emo in sorted(e for e in emotions if e != "neutral"):
        for oid in option_sig:
            chosen_intensity = peak_choice_intensity.get((oid, emo))
            if chosen_intensity is None:
                continue
            pairs: List[Tuple[Tuple[int, str], Tuple[int, str]]] = []
            for (item_id, repeat_id), (_neutral_intensity, neutral_val) in neutral_by_key.items():
                v = obs.get((emo, float(chosen_intensity), item_id, repeat_id))
                if v is None:
                    continue
                pairs.append((neutral_val, v))
            if not pairs:
                continue
            n_pairs = len(pairs)
            neutral_inds: List[int] = []
            emo_inds: List[int] = []
            deltas: List[int] = []
            n01 = 0
            n10 = 0
            for (n_opt, _), (e_opt, _) in pairs:
                n_i = 1 if n_opt == oid else 0
                e_i = 1 if e_opt == oid else 0
                neutral_inds.append(n_i)
                emo_inds.append(e_i)
                deltas.append(e_i - n_i)
                if n_i == 0 and e_i == 1:
                    n01 += 1
                elif n_i == 1 and e_i == 0:
                    n10 += 1

            neutral_rate = sum(neutral_inds) / n_pairs
            emotion_rate = sum(emo_inds) / n_pairs
            delta = emotion_rate - neutral_rate
            p = _mcnemar_exact_p(n01, n10)
            lo, hi = _bootstrap_ci_mean_delta(deltas)
            # q filled later
            option_sig[oid][emo] = SigEntry(
                n_pairs=n_pairs,
                neutral_rate=neutral_rate,
                emotion_rate=emotion_rate,
                delta=delta,
                ci_low=lo,
                ci_high=hi,
                p_value=p,
                q_value=1.0,
            )
            if not (emo == "unknown" and emotion_rate < UNKNOWN_MIN_RATIO):
                option_pvals.append((("option", oid, emo), p))

        for beh in behavior_sig:
            chosen_intensity = peak_behavior_intensity.get((beh, emo))
            if chosen_intensity is None:
                continue
            beh_pairs: List[Tuple[Tuple[int, str], Tuple[int, str]]] = []
            for (item_id, repeat_id), (_neutral_intensity, neutral_val) in neutral_by_key.items():
                v = obs.get((emo, float(chosen_intensity), item_id, repeat_id))
                if v is None:
                    continue
                beh_pairs.append((neutral_val, v))
            if not beh_pairs:
                continue
            n_pairs = len(beh_pairs)
            neutral_inds = []
            emo_inds = []
            deltas = []
            n01 = 0
            n10 = 0
            for (_, n_beh), (_, e_beh) in beh_pairs:
                n_i = 1 if n_beh == beh else 0
                e_i = 1 if e_beh == beh else 0
                neutral_inds.append(n_i)
                emo_inds.append(e_i)
                deltas.append(e_i - n_i)
                if n_i == 0 and e_i == 1:
                    n01 += 1
                elif n_i == 1 and e_i == 0:
                    n10 += 1
            neutral_rate = sum(neutral_inds) / n_pairs
            emotion_rate = sum(emo_inds) / n_pairs
            delta = emotion_rate - neutral_rate
            p = _mcnemar_exact_p(n01, n10)
            lo, hi = _bootstrap_ci_mean_delta(deltas)
            behavior_sig[beh][emo] = SigEntry(
                n_pairs=n_pairs,
                neutral_rate=neutral_rate,
                emotion_rate=emotion_rate,
                delta=delta,
                ci_low=lo,
                ci_high=hi,
                p_value=p,
                q_value=1.0,
            )
            if not (emo == "unknown" and emotion_rate < UNKNOWN_MIN_RATIO):
                behavior_pvals.append((("behavior", beh, emo), p))

    # FDR correction per game-setting: across all rows*emotions within each table.
    q_opt = _bh_fdr(option_pvals)
    q_beh = _bh_fdr(behavior_pvals)
    for oid, emo_map in option_sig.items():
        for emo, entry in list(emo_map.items()):
            q = q_opt.get(("option", oid, emo), 1.0)
            option_sig[oid][emo] = SigEntry(**{**entry.__dict__, "q_value": q})
    for beh, emo_map in behavior_sig.items():
        for emo, entry in list(emo_map.items()):
            q = q_beh.get(("behavior", beh, emo), 1.0)
            behavior_sig[beh][emo] = SigEntry(**{**entry.__dict__, "q_value": q})

    return option_sig, behavior_sig


def _format_all_emotions_cell(
    deltas: List[Tuple[str, float, float]],
    sig_by_emotion: Optional[Dict[str, SigEntry]] = None,
) -> str:
    sig_by_emotion = sig_by_emotion or {}
    parts: List[Tuple[str, float, str]] = []
    for emo, delta, ratio in deltas:
        if emo == "unknown" and ratio < UNKNOWN_MIN_RATIO:
            continue
        sig = sig_by_emotion.get(emo)
        if sig is not None:
            stars = sig.stars()
            part = f"{emo}:{sig.delta:+.3f}{stars}[{sig.ci_low:+.3f},{sig.ci_high:+.3f}]"
            parts.append((emo, sig.delta, part))
            continue
        parts.append((emo, delta, f"{emo}:{delta:+.6f}"))
    parts.sort(key=lambda x: (-x[1], x[0]))
    return "; ".join(p for _, _, p in parts)


def _format_intensity_series(vals: List[Tuple[float, float]]) -> str:
    vals = sorted(vals, key=lambda x: x[0])
    return ";".join(f"{intensity:g}:{val:+.6f}" for intensity, val in vals)
def _load_collapsed_choice_ratios(path: Path) -> Dict[str, Dict[int, float]]:
    collapsed, _ = collapse_choice_over_intensity(path, unknown_threshold=None)
    return collapsed


def _load_collapsed_behavior_ratios(path: Path) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[float]]] = {}
    collapsed, _ = collapse_behavior_over_intensity(path, unknown_threshold=None)
    return collapsed


def _impact_rows_for_choice(
    run: RunRef,
    csv_path: Path,
    *,
    option_sig: Optional[Dict[int, Dict[str, SigEntry]]] = None,
    collapsed_override: Optional[Dict[str, Dict[int, float]]] = None,
) -> Tuple[List[Dict[str, object]], bool]:
    collapsed = collapsed_override if collapsed_override is not None else _load_collapsed_choice_ratios(csv_path)
    if "neutral" not in collapsed:
        return [], True
    options = sorted({o for emo in collapsed for o in collapsed[emo]})
    emotions = sorted(collapsed)
    for emo in emotions:
        for o in options:
            collapsed[emo].setdefault(o, 0.0)
    base = collapsed["neutral"]
    rows: List[Dict[str, object]] = []
    for o in options:
        neutral_ratio = base[o]
        deltas = [(emo, collapsed[emo][o] - neutral_ratio, collapsed[emo][o]) for emo in emotions if emo != "neutral"]
        deltas = _filter_unknown_if_rare(deltas)
        if not deltas:
            continue
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        all_deltas = _format_all_emotions_cell(deltas, option_sig.get(o) if option_sig is not None else None)
        rows.append(
            {
                "task": run.task,
                "model": run.model,
                "timestamp": run.timestamp,
                "option_id": o,
                "neutral_ratio": round(neutral_ratio, 6),
                "best_emotion": best[0],
                "best_delta_vs_neutral": round(best[1], 6),
                "best_ratio": round(best[2], 6),
                "worst_emotion": worst[0],
                "worst_delta_vs_neutral": round(worst[1], 6),
                "worst_ratio": round(worst[2], 6),
                "delta_range": round(best[1] - worst[1], 6),
                "all_emotion_deltas_vs_neutral": all_deltas,
            }
        )
    return rows, False


def _impact_rows_for_behavior(
    run: RunRef,
    csv_path: Path,
    *,
    behavior_sig: Optional[Dict[str, Dict[str, SigEntry]]] = None,
    collapsed_override: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, object]], bool]:
    collapsed = collapsed_override if collapsed_override is not None else _load_collapsed_behavior_ratios(csv_path)
    if "neutral" not in collapsed:
        return [], True
    behaviors = sorted({b for emo in collapsed for b in collapsed[emo]})
    emotions = sorted(collapsed)
    for emo in emotions:
        for b in behaviors:
            collapsed[emo].setdefault(b, 0.0)
    base = collapsed["neutral"]
    rows: List[Dict[str, object]] = []
    for b in behaviors:
        neutral_ratio = base[b]
        deltas = [(emo, collapsed[emo][b] - neutral_ratio, collapsed[emo][b]) for emo in emotions if emo != "neutral"]
        deltas = _filter_unknown_if_rare(deltas)
        if not deltas:
            continue
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        all_deltas = _format_all_emotions_cell(deltas, behavior_sig.get(b) if behavior_sig is not None else None)
        rows.append(
            {
                "task": run.task,
                "model": run.model,
                "timestamp": run.timestamp,
                "behavior_label": b,
                "neutral_ratio": round(neutral_ratio, 6),
                "best_emotion": best[0],
                "best_delta_vs_neutral": round(best[1], 6),
                "best_ratio": round(best[2], 6),
                "worst_emotion": worst[0],
                "worst_delta_vs_neutral": round(worst[1], 6),
                "worst_ratio": round(worst[2], 6),
                "delta_range": round(best[1] - worst[1], 6),
                "all_emotion_deltas_vs_neutral": all_deltas,
            }
        )
    return rows, False


def _intensity_rows_for_choice(
    run: RunRef,
    csv_path: Path,
    *,
    by_intensity_override: Optional[Dict[str, Dict[float, Dict[int, float]]]] = None,
) -> Tuple[List[Dict[str, object]], bool]:
    by_intensity = (
        by_intensity_override
        if by_intensity_override is not None
        else load_choice_by_intensity(csv_path, unknown_threshold=None)[0]
    )
    if "neutral" not in by_intensity:
        return [], True

    # baseline neutral per option = mean over all neutral intensities present
    neutral_ints = sorted(by_intensity["neutral"])
    options = sorted({o for emo in by_intensity for i in by_intensity[emo] for o in by_intensity[emo][i]})
    for i in neutral_ints:
        for o in options:
            by_intensity["neutral"][i].setdefault(o, 0.0)
    neutral_base: Dict[int, float] = {o: mean([by_intensity["neutral"][i][o] for i in neutral_ints]) for o in options}

    rows: List[Dict[str, object]] = []
    for emo in sorted(e for e in by_intensity if e != "neutral"):
        emo_ints = sorted(by_intensity[emo])
        for i in emo_ints:
            for o in options:
                by_intensity[emo][i].setdefault(o, 0.0)

        for o in options:
            neutral_ratio = neutral_base[o]
            deltas = [(i, by_intensity[emo][i][o] - neutral_ratio) for i in emo_ints]
            ratios = [(i, by_intensity[emo][i][o]) for i in emo_ints]
            best_i, best_delta = max(deltas, key=lambda x: x[1])
            worst_i, worst_delta = min(deltas, key=lambda x: x[1])
            best_ratio = by_intensity[emo][best_i][o]
            worst_ratio = by_intensity[emo][worst_i][o]
            rows.append(
                {
                    "task": run.task,
                    "model": run.model,
                    "timestamp": run.timestamp,
                    "option_id": o,
                    "emotion": emo,
                    "neutral_ratio": round(neutral_ratio, 6),
                    "best_intensity": best_i,
                    "best_delta_vs_neutral": round(best_delta, 6),
                    "best_ratio": round(best_ratio, 6),
                    "worst_intensity": worst_i,
                    "worst_delta_vs_neutral": round(worst_delta, 6),
                    "worst_ratio": round(worst_ratio, 6),
                    "delta_range_across_intensity": round(best_delta - worst_delta, 6),
                    "deltas_by_intensity": _format_intensity_series(deltas),
                    "ratios_by_intensity": _format_intensity_series([(i, r) for i, r in ratios]),
                }
            )
    return rows, False


def _intensity_rows_for_behavior(
    run: RunRef,
    csv_path: Path,
    *,
    by_intensity_override: Optional[Dict[str, Dict[float, Dict[str, float]]]] = None,
) -> Tuple[List[Dict[str, object]], bool]:
    by_intensity = (
        by_intensity_override
        if by_intensity_override is not None
        else load_behavior_by_intensity(csv_path, unknown_threshold=None)[0]
    )
    if "neutral" not in by_intensity:
        return [], True

    neutral_ints = sorted(by_intensity["neutral"])
    behaviors = sorted({b for emo in by_intensity for i in by_intensity[emo] for b in by_intensity[emo][i]})
    for i in neutral_ints:
        for b in behaviors:
            by_intensity["neutral"][i].setdefault(b, 0.0)
    neutral_base: Dict[str, float] = {b: mean([by_intensity["neutral"][i][b] for i in neutral_ints]) for b in behaviors}

    rows: List[Dict[str, object]] = []
    for emo in sorted(e for e in by_intensity if e != "neutral"):
        emo_ints = sorted(by_intensity[emo])
        for i in emo_ints:
            for b in behaviors:
                by_intensity[emo][i].setdefault(b, 0.0)

        for b in behaviors:
            neutral_ratio = neutral_base[b]
            deltas = [(i, by_intensity[emo][i][b] - neutral_ratio) for i in emo_ints]
            ratios = [(i, by_intensity[emo][i][b]) for i in emo_ints]
            best_i, best_delta = max(deltas, key=lambda x: x[1])
            worst_i, worst_delta = min(deltas, key=lambda x: x[1])
            best_ratio = by_intensity[emo][best_i][b]
            worst_ratio = by_intensity[emo][worst_i][b]
            rows.append(
                {
                    "task": run.task,
                    "model": run.model,
                    "timestamp": run.timestamp,
                    "behavior_label": b,
                    "emotion": emo,
                    "neutral_ratio": round(neutral_ratio, 6),
                    "best_intensity": best_i,
                    "best_delta_vs_neutral": round(best_delta, 6),
                    "best_ratio": round(best_ratio, 6),
                    "worst_intensity": worst_i,
                    "worst_delta_vs_neutral": round(worst_delta, 6),
                    "worst_ratio": round(worst_ratio, 6),
                    "delta_range_across_intensity": round(best_delta - worst_delta, 6),
                    "deltas_by_intensity": _format_intensity_series(deltas),
                    "ratios_by_intensity": _format_intensity_series([(i, r) for i, r in ratios]),
                }
            )
    return rows, False


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_str(value: object) -> str:
    return str(value)


def _as_float(value: object) -> float:
    return float(str(value))


def _render_delta_heatmap(
    *,
    rows: List[Dict[str, object]],
    value_key: str,
    column_key: str,
    title: str,
    out_path: Path,
) -> None:
    if not rows:
        return

    row_labels = sorted({f"{str(row['task'])} | {str(row['model'])}" for row in rows})
    col_labels = sorted({str(row[column_key]) for row in rows})
    values: Dict[Tuple[str, str], float] = {}
    for row in rows:
        row_label = f"{str(row['task'])} | {str(row['model'])}"
        col_label = str(row[column_key])
        values[(row_label, col_label)] = _as_float(row[value_key])

    matrix = [[values.get((row_label, col_label), 0.0) for col_label in col_labels] for row_label in row_labels]

    height = max(4.0, 0.35 * len(row_labels) + 1.5)
    width = max(6.0, 1.2 * len(col_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    vmax = max(abs(v) for row in matrix for v in row) if matrix and matrix[0] else 1.0
    if vmax == 0.0:
        vmax = 1.0
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for row_idx, row_vals in enumerate(matrix):
        for col_idx, value in enumerate(row_vals):
            ax.text(col_idx, row_idx, f"{value:+.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Delta vs neutral")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_markdown(
    *,
    root: Path,
    runs: List[RunRef],
    option_csv: Path,
    behavior_csv: Optional[Path],
    option_intensity_csv: Path,
    behavior_intensity_csv: Optional[Path],
    option_rows: List[Dict[str, object]],
    behavior_rows: List[Dict[str, object]],
    option_intensity_rows: List[Dict[str, object]],
    behavior_intensity_rows: List[Dict[str, object]],
    skipped_missing_neutral: List[Path],
    top_n: int,
    per_game_n: int,
    heatmaps_dir: Optional[Path],
) -> str:
    lines: List[str] = []
    lines.append("# Game-Theory Decision Impact Report (vs neutral)")
    lines.append("")
    lines.append("## Data Used")
    lines.append(f"- Root scanned: `{root}`")
    lines.append("- Input files searched: `**/summary_choice_ratio.csv`, `**/summary_behavior_ratio.csv`")
    lines.append(f"- Latest run per (model, game_setting): {len(runs)}")
    for run in runs:
        lines.append(f"  - `{run.dir_path}`")
    lines.append("")
    lines.append("## Method")
    lines.append("- For each `(model, game_setting)`, select the latest timestamped run directory.")
    lines.append("- Collapse intensity by averaging `ratio` over all intensities present.")
    lines.append("- Compute `delta_vs_neutral = ratio(emotion) - ratio(neutral)` for each option/behavior.")
    lines.append("- Summarize best/worst emotion deltas and `delta_range = best - worst`.")
    lines.append("- Additionally, compute per-emotion `delta_range_across_intensity` with per-intensity deltas (vs neutral mean).")
    lines.append("- In per-game tables, show deltas for all emotions (vs neutral), ranked by Δ descending.")
    lines.append("- When `raw_results.json` is available, annotate each emotion as `emo:Δ{sig}[ci_low,ci_high]`.")
    lines.append("  - `{sig}` is `! / !! / !!!` based on Benjamini–Hochberg FDR per game-setting (within Option table / Behavior table).")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- Option CSV: `{option_csv}`")
    if behavior_csv is not None:
        lines.append(f"- Behavior CSV: `{behavior_csv}`")
    else:
        lines.append("- Behavior CSV: (not generated)")
    lines.append(f"- Option-by-intensity CSV: `{option_intensity_csv}`")
    if behavior_intensity_csv is not None:
        lines.append(f"- Behavior-by-intensity CSV: `{behavior_intensity_csv}`")
    else:
        lines.append("- Behavior-by-intensity CSV: (not generated)")
    if heatmaps_dir is not None:
        lines.append(f"- Heatmaps: `{heatmaps_dir}`")
    lines.append("")

    def _table(rows: List[Dict[str, object]], *, key_col: str, label: str) -> None:
        lines.append(f"## Strongest {label} Effects (Top {top_n} by delta_range)")
        lines.append(f"| game_setting | model | {key_col} | neutral | best (Δ) | worst (Δ) | range |")
        lines.append("|---|---|---|---:|---|---|---:|")
        top = sorted(rows, key=lambda r: _as_float(r["delta_range"]), reverse=True)[:top_n]
        for r in top:
            lines.append(
                "| {task} | {model} | {k} | {neutral:.3f} | {best} ({best_delta:+.3f}) | {worst} ({worst_delta:+.3f}) | {rng:.3f} |".format(
                    task=_as_str(r["task"]),
                    model=_as_str(r["model"]),
                    k=_as_str(r[key_col]),
                    neutral=_as_float(r["neutral_ratio"]),
                    best=_as_str(r["best_emotion"]),
                    best_delta=_as_float(r["best_delta_vs_neutral"]),
                    worst=_as_str(r["worst_emotion"]),
                    worst_delta=_as_float(r["worst_delta_vs_neutral"]),
                    rng=_as_float(r["delta_range"]),
                )
            )
        lines.append("")

        if per_game_n > 0:
            lines.append(f"## Per Game Setting {label} (Top {per_game_n} by delta_range)")
        else:
            lines.append(f"## Per Game Setting {label} (All models)")
        tasks = sorted({_as_str(r["task"]) for r in rows})
        for task in tasks:
            lines.append(f"### {task}")
            lines.append(f"| model | {key_col} | neutral | all emotion deltas (Δ vs neutral) | range |")
            lines.append("|---|---|---:|---|---:|")
            task_rows = [r for r in rows if _as_str(r["task"]) == task]
            if per_game_n > 0:
                task_rows = sorted(task_rows, key=lambda r: _as_float(r["delta_range"]), reverse=True)[:per_game_n]
            else:
                task_rows = sorted(task_rows, key=lambda r: (str(r["model"]), str(r[key_col])))
            for r in task_rows:
                lines.append(
                    "| {model} | {k} | {neutral:.3f} | {deltas} | {rng:.3f} |".format(
                        model=_as_str(r["model"]),
                        k=_as_str(r[key_col]),
                        neutral=_as_float(r["neutral_ratio"]),
                        deltas=str(r.get("all_emotion_deltas_vs_neutral", "")),
                        rng=_as_float(r["delta_range"]),
                    )
                )
            lines.append("")

            # Item change analysis (behavior only): share of paired items that flip vs neutral.
            if label == "Behavior":
                # Use latest run dir per model+task; render one line per model for compactness.
                model_to_run = {run.model: run for run in runs if run.task == task}
                if model_to_run:
                    lines.append("#### Item Change vs Neutral (paired by item_id, ignores intensity)")
                    lines.append("| model | change rates (emotion:%, n) |")
                    lines.append("|---|---|")
                    for model in sorted(model_to_run):
                        rates = _item_change_rates_for_run(model_to_run[model])
                        parts: List[Tuple[float, str]] = []
                        for emo, (rate, n) in rates.items():
                            if emo == "unknown" and rate < UNKNOWN_MIN_RATIO:
                                continue
                            parts.append((rate, f"{emo}:{rate*100:.1f}% (n={n})"))
                        parts.sort(key=lambda x: (-x[0], x[1]))
                        cell = "; ".join(p for _, p in parts)
                        lines.append(f"| {model} | {cell} |")
                    lines.append("")
    _table(option_rows, key_col="option_id", label="Option")

    if behavior_rows:
        _table(behavior_rows, key_col="behavior_label", label="Behavior")
    else:
        lines.append("## Behavior Effects")
        lines.append("No behavior ratio inputs found (`summary_behavior_ratio.csv`).")
        lines.append("")

    def _intensity_table(rows: List[Dict[str, object]], *, key_col: str, label: str) -> None:
        lines.append(f"## {label} Intensity Sensitivity (Top {top_n} by delta_range_across_intensity)")
        lines.append(f"| game_setting | model | emotion | {key_col} | best (intensity, Δ) | worst (intensity, Δ) | range |")
        lines.append("|---|---|---|---|---|---|---:|")
        top = sorted(rows, key=lambda r: _as_float(r["delta_range_across_intensity"]), reverse=True)[:top_n]
        for r in top:
            lines.append(
                "| {task} | {model} | {emo} | {k} | {best_i:g} ({best_delta:+.3f}) | {worst_i:g} ({worst_delta:+.3f}) | {rng:.3f} |".format(
                    task=_as_str(r["task"]),
                    model=_as_str(r["model"]),
                    emo=_as_str(r["emotion"]),
                    k=_as_str(r[key_col]),
                    best_i=_as_float(r["best_intensity"]),
                    best_delta=_as_float(r["best_delta_vs_neutral"]),
                    worst_i=_as_float(r["worst_intensity"]),
                    worst_delta=_as_float(r["worst_delta_vs_neutral"]),
                    rng=_as_float(r["delta_range_across_intensity"]),
                )
            )
        lines.append("")

    if option_intensity_rows:
        _intensity_table(option_intensity_rows, key_col="option_id", label="Option")
    if behavior_intensity_rows:
        _intensity_table(behavior_intensity_rows, key_col="behavior_label", label="Behavior")

    if skipped_missing_neutral:
        lines.append("## Skipped runs (missing neutral)")
        for p in sorted({p.parent for p in skipped_missing_neutral}):
            lines.append(f"- `{p}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_game_theory_impact_report(
    *,
    root: Path,
    out_dir: Optional[Path] = None,
    top_n: int = 20,
    per_game_n: int = 0,
    write_heatmaps: bool = False,
    unknown_threshold: Optional[float] = None,
    heatmap_norm: str = "symlog",
    heatmap_symlog_linthresh: float = 0.01,
) -> ReportOutputs:
    runs = _discover_latest_runs(root)
    if not runs:
        raise ValueError(f"No timestamped run directories found under {root}")

    option_rows: List[Dict[str, object]] = []
    behavior_rows: List[Dict[str, object]] = []
    option_intensity_rows: List[Dict[str, object]] = []
    behavior_intensity_rows: List[Dict[str, object]] = []
    skipped_missing_neutral: List[Path] = []
    for run in runs:
        option_sig, behavior_sig = _sig_maps_for_run(run, unknown_threshold=unknown_threshold)
        choice_csv = run.dir_path / "summary_choice_ratio.csv"
        behavior_csv = run.dir_path / "summary_behavior_ratio.csv"
        if choice_csv.exists():
            collapsed, _ = collapse_choice_over_intensity(choice_csv, unknown_threshold=unknown_threshold)
            by_intensity, _ = load_choice_by_intensity(choice_csv, unknown_threshold=unknown_threshold)
            rows, missing_neutral = _impact_rows_for_choice(
                run, choice_csv, option_sig=option_sig or None, collapsed_override=collapsed
            )
            option_rows.extend(rows)
            if missing_neutral:
                skipped_missing_neutral.append(choice_csv)
            rows, missing_neutral = _intensity_rows_for_choice(run, choice_csv, by_intensity_override=by_intensity)
            option_intensity_rows.extend(rows)
            if missing_neutral:
                skipped_missing_neutral.append(choice_csv)
        if behavior_csv.exists():
            collapsed_b, _ = collapse_behavior_over_intensity(behavior_csv, unknown_threshold=unknown_threshold)
            by_intensity_b, _ = load_behavior_by_intensity(behavior_csv, unknown_threshold=unknown_threshold)
            rows, missing_neutral = _impact_rows_for_behavior(
                run, behavior_csv, behavior_sig=behavior_sig or None, collapsed_override=collapsed_b
            )
            behavior_rows.extend(rows)
            if missing_neutral:
                skipped_missing_neutral.append(behavior_csv)
            rows, missing_neutral = _intensity_rows_for_behavior(run, behavior_csv, by_intensity_override=by_intensity_b)
            behavior_intensity_rows.extend(rows)
            if missing_neutral:
                skipped_missing_neutral.append(behavior_csv)

    if not option_rows:
        raise ValueError(f"No usable summary_choice_ratio.csv found under {root} (need neutral)")

    out_base = out_dir if out_dir is not None else root
    option_out = out_base / "option_impacted_by_emo_vs_neutral_latest.csv"
    behavior_out = out_base / "behavior_impacted_emo_vs_neutral_latest.csv" if behavior_rows else None
    option_intensity_out = out_base / "option_intensity_impacted_by_emo_vs_neutral_latest.csv"
    behavior_intensity_out = (
        out_base / "behavior_intensity_impacted_emo_vs_neutral_latest.csv" if behavior_intensity_rows else None
    )
    heatmaps_dir = out_base / "heatmaps" if write_heatmaps else None
    report_out = out_base / "game_theory_impact_report.md"

    _write_csv(option_rows, option_out)
    if behavior_out is not None:
        _write_csv(behavior_rows, behavior_out)
    _write_csv(option_intensity_rows, option_intensity_out)
    if behavior_intensity_out is not None:
        _write_csv(behavior_intensity_rows, behavior_intensity_out)

    if heatmaps_dir is not None:
        by_task: Dict[str, Dict[str, Path]] = {}
        for run in runs:
            behavior_csv = run.dir_path / "summary_behavior_ratio.csv"
            if behavior_csv.exists():
                by_task.setdefault(run.task, {})[run.model] = behavior_csv
        for task, model_map in sorted(by_task.items()):
            write_behavior_change_heatmap_pdf(
                task=task,
                model_to_behavior_csv=model_map,
                out_dir=heatmaps_dir,
                unknown_threshold=unknown_threshold,
                heatmap_norm=heatmap_norm,
                symlog_linthresh=heatmap_symlog_linthresh,
            )

    report_out.write_text(
        _render_markdown(
            root=root,
            runs=runs,
            option_csv=option_out,
            behavior_csv=behavior_out,
            option_intensity_csv=option_intensity_out,
            behavior_intensity_csv=behavior_intensity_out,
            option_rows=option_rows,
            behavior_rows=behavior_rows,
            option_intensity_rows=option_intensity_rows,
            behavior_intensity_rows=behavior_intensity_rows,
            skipped_missing_neutral=skipped_missing_neutral,
            top_n=top_n,
            per_game_n=per_game_n,
            heatmaps_dir=heatmaps_dir,
        ),
        encoding="utf-8",
    )

    return ReportOutputs(
        option_csv_path=option_out,
        behavior_csv_path=behavior_out,
        option_intensity_csv_path=option_intensity_out,
        behavior_intensity_csv_path=behavior_intensity_out,
        heatmaps_dir=heatmaps_dir,
        report_path=report_out,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/new_game_theory_decision/shuffle_choices"),
        help="Directory containing timestamped run subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to write outputs (default: same as --root). Useful when --root is a symlinked read-only location.",
    )
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--per_game_n", type=int, default=0, help="0 means include all models")
    parser.add_argument("--write_heatmaps", action="store_true", help="Write behavior-change heatmap PDFs under out_dir/heatmaps")
    parser.add_argument(
        "--unknown_threshold",
        type=float,
        default=None,
        help="If set, drop (emotion,intensity) slices with unknown ratio > threshold (behavior: behavior=='unknown'; choice: option_id==-1).",
    )
    parser.add_argument(
        "--heatmap_norm",
        type=str,
        default="symlog",
        choices=["linear", "symlog"],
        help="Heatmap normalization (only used with --write_heatmaps).",
    )
    parser.add_argument(
        "--heatmap_symlog_linthresh",
        type=float,
        default=0.01,
        help="Symlog linthresh for heatmaps (only used when --heatmap_norm symlog).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    generate_game_theory_impact_report(
        root=args.root,
        out_dir=args.out_dir,
        top_n=args.top_n,
        per_game_n=args.per_game_n,
        write_heatmaps=args.write_heatmaps,
        unknown_threshold=args.unknown_threshold,
        heatmap_norm=args.heatmap_norm,
        heatmap_symlog_linthresh=args.heatmap_symlog_linthresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
