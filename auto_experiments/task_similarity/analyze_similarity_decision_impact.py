"""
Responsible: auto_experiments/task_similarity/analyze_similarity_decision_impact.py
Purpose: Analyze how delta-activation similarity (cos(Δ^anger, Δ^pd)) relates to
         the model's final Prisoner's Dilemma decision.

This script *does not* recompute deltas. It consumes:
1) A similarity run directory produced by `emotion_pd_delta_similarity.py`:
   - metadata.json (intensities, item_ids, controlled_layers, ...)
   - cosines.npy    shape (n_int, n_samples, n_layers)  [emotion vs PD-defect]
   - cosines_pd_cooperate.npy (optional)                [emotion vs PD-cooperate]
   - pref_cosines.npy (optional)                        [cos_defect - cos_cooperate]
2) An experiment result directory containing `detailed_results.csv` from EmotionExperiment:
   - item_id, emotion, intensity, chosen_behavior

Outputs (to --out_dir):
- joined_rows.csv: one row per (item_id, intensity, layer) with cosine + decision label
- layer_impact_summary.csv: per (intensity, layer) association stats (r + mean-diff)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, cast

import numpy as np


@dataclass(frozen=True)
class SimilarityRun:
    item_ids: List[int]
    intensities: List[float]
    controlled_layers: List[int]
    measurement_layers: List[int]
    cosines: np.ndarray  # (n_int, n_samples, n_layers)
    cosines_pd_cooperate: np.ndarray | None  # (n_int, n_samples, n_layers)
    pref_cosines: np.ndarray | None  # (n_int, n_samples, n_layers)


def load_similarity_run(run_dir: Path) -> SimilarityRun:
    meta = json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))
    cos = np.load(Path(run_dir) / "cosines.npy")
    coop_path = Path(run_dir) / "cosines_pd_cooperate.npy"
    pref_path = Path(run_dir) / "pref_cosines.npy"
    cos_coop = np.load(coop_path) if coop_path.exists() else None
    pref = np.load(pref_path) if pref_path.exists() else None
    item_ids = [int(x) for x in meta["item_ids"]]
    intensities = [float(x) for x in meta["intensities"]]
    controlled_layers = [int(x) for x in meta["controlled_layers"]]
    measurement_layers = [int(x) for x in meta["measurement_layers"]]
    if cos.ndim != 3:
        raise ValueError(f"cosines must be 3D (n_int,n_samples,n_layers), got {cos.shape}")
    if cos.shape[0] != len(intensities):
        raise ValueError("cosines first dim must match intensities")
    if cos.shape[1] != len(item_ids):
        raise ValueError("cosines second dim must match item_ids")
    if cos.shape[2] != len(measurement_layers):
        raise ValueError("cosines third dim must match measurement_layers")
    if cos_coop is not None:
        if cos_coop.shape != cos.shape:
            raise ValueError(f"cosines_pd_cooperate must match cosines shape, got {cos_coop.shape} vs {cos.shape}")
    if pref is not None:
        if pref.shape != cos.shape:
            raise ValueError(f"pref_cosines must match cosines shape, got {pref.shape} vs {cos.shape}")
    return SimilarityRun(
        item_ids=item_ids,
        intensities=intensities,
        controlled_layers=controlled_layers,
        measurement_layers=measurement_layers,
        cosines=cos.astype(np.float32, copy=False),
        cosines_pd_cooperate=cos_coop.astype(np.float32, copy=False) if cos_coop is not None else None,
        pref_cosines=pref.astype(np.float32, copy=False) if pref is not None else None,
    )


def load_pd_decisions_from_detailed_results(
    detailed_results_csv: Path, *, emotion: str = "anger"
) -> Dict[Tuple[int, float], str]:
    """
    Returns mapping (item_id, intensity) -> chosen_behavior ('defect' or 'cooperate').
    Filters to the given emotion only.
    """
    out: Dict[Tuple[int, float], str] = {}
    with Path(detailed_results_csv).open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"emotion", "intensity", "item_id", "chosen_behavior"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {detailed_results_csv}: {sorted(missing)}")
        for row in r:
            if str(row["emotion"]) != str(emotion):
                continue
            item_id = int(float(row["item_id"]))
            intensity = float(row["intensity"])
            behavior = str(row["chosen_behavior"]).strip().lower()
            if behavior not in {"defect", "cooperate"}:
                continue
            out[(item_id, intensity)] = behavior
    return out


def load_prompts_from_raw_results(
    raw_results_json: Path, *, emotion: str = "anger"
) -> Dict[Tuple[int, float], Dict[str, str]]:
    """
    Returns mapping (item_id, intensity) -> {"prompt": str, "response": str}.
    Filters to the given emotion only.
    """
    raw = json.loads(Path(raw_results_json).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"raw_results.json must be a list of records: {raw_results_json}")

    out: Dict[Tuple[int, float], Dict[str, str]] = {}
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("emotion")) != str(emotion):
            continue
        if "item_id" not in rec or "intensity" not in rec:
            continue
        item_id = int(float(rec["item_id"]))
        intensity = float(rec["intensity"])
        prompt = rec.get("prompt")
        resp = rec.get("response")
        if not isinstance(prompt, str):
            prompt = ""
        if not isinstance(resp, str):
            resp = ""
        out[(item_id, intensity)] = {"prompt": prompt, "response": resp}
    return out


def join_similarity_with_decisions(
    sim: SimilarityRun, decisions: Mapping[Tuple[int, float], str]
) -> Dict[Tuple[int, float], Dict[str, object]]:
    """
    Join by (item_id,intensity). Output holds:
      - defect: 1 if behavior=='defect' else 0
      - behavior: original string
      - cosine: np.ndarray shape (n_layers,) for that sample+intensity (emotion vs PD-defect)
      - cosine_pd_cooperate: optional np.ndarray shape (n_layers,)
      - pref_cosine: optional np.ndarray shape (n_layers,)
    """
    out: Dict[Tuple[int, float], Dict[str, object]] = {}
    id_to_idx = {int(item_id): idx for idx, item_id in enumerate(sim.item_ids)}
    int_to_idx = {float(v): i for i, v in enumerate(sim.intensities)}

    for (item_id, intensity), behavior in decisions.items():
        if item_id not in id_to_idx:
            continue
        if float(intensity) not in int_to_idx:
            continue
        i_s = id_to_idx[item_id]
        i_int = int_to_idx[float(intensity)]
        cos = sim.cosines[i_int, i_s, :]
        cos_coop = sim.cosines_pd_cooperate[i_int, i_s, :] if sim.cosines_pd_cooperate is not None else None
        pref = sim.pref_cosines[i_int, i_s, :] if sim.pref_cosines is not None else None
        out[(item_id, float(intensity))] = {
            "behavior": behavior,
            "defect": 1 if behavior == "defect" else 0,
            "cosine": cos,
            "cosine_pd_cooperate": cos_coop,
            "pref_cosine": pref,
        }
    return out


def _pearsonr_binary(x: np.ndarray, y01: np.ndarray) -> float:
    """
    Pearson correlation between x and a binary label y in {0,1}.
    Returns NaN if variance is zero or not enough samples.
    """
    if x.size != y01.size:
        raise ValueError("x and y must have same size")
    if x.size < 3:
        return float("nan")
    x = x.astype(np.float64, copy=False)
    y = y01.astype(np.float64, copy=False)
    vx = float(np.var(x))
    vy = float(np.var(y))
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze decision impact of delta-activation similarity (cos(Δ^anger, Δ^pd))."
    )
    p.add_argument(
        "--similarity_run_dir",
        required=True,
        help="Directory produced by emotion_pd_delta_similarity.py (contains metadata.json + cosines.npy).",
    )
    p.add_argument(
        "--result_dir",
        required=True,
        help="Experiment results directory containing detailed_results.csv (and raw_results.json).",
    )
    p.add_argument("--emotion", default="anger")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <similarity_run_dir>/decision_impact/<emotion>).",
    )
    args = p.parse_args()

    sim_dir = Path(args.similarity_run_dir)
    result_dir = Path(args.result_dir)
    detailed = result_dir / "detailed_results.csv"
    if not detailed.exists():
        raise FileNotFoundError(f"Missing detailed_results.csv: {detailed}")
    raw_results = result_dir / "raw_results.json"
    raw_payload = None
    if raw_results.exists():
        raw_payload = load_prompts_from_raw_results(raw_results, emotion=str(args.emotion))

    sim = load_similarity_run(sim_dir)
    decisions = load_pd_decisions_from_detailed_results(detailed, emotion=str(args.emotion))
    joined = join_similarity_with_decisions(sim, decisions)

    out_dir = Path(args.out_dir) if args.out_dir else (sim_dir / "decision_impact" / str(args.emotion))
    out_dir.mkdir(parents=True, exist_ok=True)

    controlled = set(int(x) for x in sim.controlled_layers)

    # Per-sample info with text (prompt/response) when available.
    samples_path = out_dir / "samples_with_decision.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "intensity", "behavior", "defect", "prompt", "response"])
        for (item_id, intensity), payload in sorted(joined.items()):
            prompt = ""
            resp = ""
            if raw_payload is not None:
                pr = raw_payload.get((int(item_id), float(intensity)))
                if pr:
                    prompt = pr.get("prompt", "")
                    resp = pr.get("response", "")
            w.writerow(
                [
                    int(item_id),
                    float(intensity),
                    str(payload["behavior"]),
                    int(cast(int, payload["defect"])),
                    prompt,
                    resp,
                ]
            )

    # Row-level joined output (layer-wise).
    joined_path = out_dir / "joined_rows.csv"
    with joined_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item_id",
                "intensity",
                "layer",
                "controlled",
                "behavior",
                "defect",
                "cosine",
                "cosine_pd_cooperate",
                "pref_cosine",
            ]
        )
        for (item_id, intensity), payload in sorted(joined.items()):
            cos = np.asarray(payload["cosine"], dtype=np.float32)
            cos_coop = payload.get("cosine_pd_cooperate")
            pref = payload.get("pref_cosine")
            cos_coop_arr = np.asarray(cos_coop, dtype=np.float32) if cos_coop is not None else None
            pref_arr = np.asarray(pref, dtype=np.float32) if pref is not None else None
            for layer_idx, layer in enumerate(sim.measurement_layers):
                w.writerow(
                    [
                        int(item_id),
                        float(intensity),
                        int(layer),
                        1 if int(layer) in controlled else 0,
                        str(payload["behavior"]),
                        int(cast(int, payload["defect"])),
                        float(cos[layer_idx]),
                        float(cos_coop_arr[layer_idx]) if cos_coop_arr is not None else float("nan"),
                        float(pref_arr[layer_idx]) if pref_arr is not None else float("nan"),
                    ]
                )

    # Summary stats per (intensity, layer).
    summary_path = out_dir / "layer_impact_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "intensity",
                "layer",
                "controlled",
                "n",
                "n_defect",
                "n_cooperate",
                "mean_cos_defect",
                "mean_cos_cooperate",
                "mean_diff_defect_minus_coop",
                "pearson_r(defect,cosine)",
                "mean_pref_defect",
                "mean_pref_cooperate",
                "mean_pref_diff_defect_minus_coop",
                "pearson_r(defect,pref_cosine)",
            ]
        )

        for intensity in sim.intensities:
            rows = [(k, v) for k, v in joined.items() if k[1] == float(intensity)]
            if not rows:
                continue
            y = np.array([int(cast(int, v["defect"])) for _, v in rows], dtype=np.int64)
            for layer_idx, layer in enumerate(sim.measurement_layers):
                x = np.array(
                    [float(np.asarray(cast(np.ndarray, v["cosine"]), dtype=np.float64)[layer_idx]) for _, v in rows],
                    dtype=np.float64,
                )
                # ignore NaNs
                m = ~np.isnan(x)
                x = x[m]
                yy = y[m]
                if x.size == 0:
                    continue
                n = int(x.size)
                n_def = int(np.sum(yy == 1))
                n_coop = int(np.sum(yy == 0))
                mean_def = float(np.mean(x[yy == 1])) if n_def else float("nan")
                mean_coop = float(np.mean(x[yy == 0])) if n_coop else float("nan")
                mean_diff = mean_def - mean_coop if (n_def and n_coop) else float("nan")
                r = _pearsonr_binary(x.astype(np.float64), yy.astype(np.int64))

                pref_vals: List[float] = []
                pref_y: List[int] = []
                for _, payload in rows:
                    pref_vec = payload.get("pref_cosine")
                    if pref_vec is None:
                        continue
                    pref_vals.append(float(np.asarray(pref_vec, dtype=np.float64)[layer_idx]))
                    pref_y.append(int(cast(int, payload["defect"])))
                pref_x = np.asarray(pref_vals, dtype=np.float64)
                pref_y01 = np.asarray(pref_y, dtype=np.int64)
                m_pref = ~np.isnan(pref_x)
                pref_x2 = pref_x[m_pref]
                pref_y2 = pref_y01[m_pref]
                n_pref_def = int(np.sum(pref_y2 == 1))
                n_pref_coop = int(np.sum(pref_y2 == 0))
                mean_pref_def = float(np.mean(pref_x2[pref_y2 == 1])) if n_pref_def else float("nan")
                mean_pref_coop = float(np.mean(pref_x2[pref_y2 == 0])) if n_pref_coop else float("nan")
                mean_pref_diff = mean_pref_def - mean_pref_coop if (n_pref_def and n_pref_coop) else float("nan")
                r_pref = _pearsonr_binary(pref_x2.astype(np.float64), pref_y2.astype(np.int64))
                w.writerow(
                    [
                        float(intensity),
                        int(layer),
                        1 if int(layer) in controlled else 0,
                        n,
                        n_def,
                        n_coop,
                        mean_def,
                        mean_coop,
                        mean_diff,
                        r,
                        mean_pref_def,
                        mean_pref_coop,
                        mean_pref_diff,
                        r_pref,
                    ]
                )

    meta = {
        "similarity_run_dir": str(sim_dir),
        "result_dir": str(result_dir),
        "emotion": str(args.emotion),
        "n_joined_pairs": len(joined),
        "has_raw_results": bool(raw_payload is not None),
        "note": "Joined on (item_id,intensity); prompt formatting may differ if result_dir used shuffle_options.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
