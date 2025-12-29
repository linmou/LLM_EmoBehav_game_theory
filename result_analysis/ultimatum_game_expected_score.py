"""
result_analysis/ultimatum_game_expected_score.py

Ultimatum Game expected-score deltas vs neutral for:
- Proposer (offer_low/offer_medium/offer_high)
- Responder (reject/accept)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd  # type: ignore[import-untyped]


@dataclass(frozen=True)
class UltimatumGameSpec:
    benchmark_substring: str
    output_prefix: str
    title_role: str
    decision_encoding_md: str
    up_label: str
    down_label: str
    behavior_to_score: dict[str, int]


PROPOSER_SPEC = UltimatumGameSpec(
    benchmark_substring="Ultimatum_Game_Proposer",
    output_prefix="ultimatum_proposer",
    title_role="Proposer",
    decision_encoding_md="Decision encoding used: `offer_low=0`, `offer_medium=1`, `offer_high=2`.",
    up_label="higher offer",
    down_label="lower offer",
    behavior_to_score={"offer_low": 0, "offer_medium": 1, "offer_high": 2},
)

RESPONDER_SPEC = UltimatumGameSpec(
    benchmark_substring="Ultimatum_Game_Responder",
    output_prefix="ultimatum_responder",
    title_role="Responder",
    decision_encoding_md="Decision encoding used: `reject=0`, `accept=1`.",
    up_label="more accept",
    down_label="more reject",
    behavior_to_score={"reject": 0, "accept": 1},
)


@dataclass(frozen=True)
class AnalysisOutputs:
    item_expected_score_deltas_path: str
    item_max_delta_summary_path: str
    aggregate_by_emotion_intensity_path: str
    aggregate_by_emotion_path: str
    report_md_path: str
    item_expected_score_deltas: pd.DataFrame
    item_max_delta_summary: pd.DataFrame
    aggregate_by_emotion_intensity: pd.DataFrame
    aggregate_by_emotion: pd.DataFrame


def _infer_model_name(exp: dict[str, Any]) -> str:
    model_name = exp.get("model_name")
    if isinstance(model_name, str) and model_name:
        return Path(model_name).name
    output_dir = exp.get("output_dir")
    if isinstance(output_dir, str) and output_dir:
        return output_dir.split("_game_theory_")[0].split("/")[-1]
    return "unknown_model"


def _iter_run_dirs(report: dict[str, Any], spec: UltimatumGameSpec) -> Iterable[tuple[str, Path]]:
    experiments = report.get("experiments", {})
    if not isinstance(experiments, dict):
        return

    for exp in experiments.values():
        if not isinstance(exp, dict):
            continue
        status = exp.get("status")
        if isinstance(status, str) and status.lower() != "completed":
            continue
        benchmark_name = exp.get("benchmark_name", "")
        if not isinstance(benchmark_name, str) or spec.benchmark_substring not in benchmark_name:
            continue
        out = exp.get("output_dir")
        if not isinstance(out, str) or not out:
            continue
        model = _infer_model_name(exp)
        yield model, Path(out)


def _behavior_from_record(rec: dict[str, Any]) -> str | None:
    score = rec.get("score")
    if score is None:
        return None
    try:
        option_id = int(float(score))
    except Exception:
        return None
    options = (rec.get("metadata", {}).get("item_metadata", {}).get("options", []))
    if not isinstance(options, list):
        return None
    for opt in options:
        if not isinstance(opt, dict):
            continue
        if int(opt.get("id", -1)) == option_id:
            return opt.get("behavior")
    return None


def _load_records(run_dir: Path, model: str, spec: UltimatumGameSpec) -> list[dict[str, Any]]:
    raw_path = run_dir / "raw_results.json"
    data = json.loads(raw_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {raw_path}")

    rows: list[dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        if rec.get("error"):
            continue
        intensity = rec.get("intensity")
        if intensity is None:
            continue
        try:
            intensity_f = float(intensity)
        except Exception:
            continue
        behavior = _behavior_from_record(rec)
        if behavior not in spec.behavior_to_score:
            continue
        rows.append(
            {
                "model": model,
                "item_id": rec.get("item_id"),
                "emotion": rec.get("emotion"),
                "intensity": intensity_f,
                "behavior": behavior,
                "decision_score": spec.behavior_to_score[behavior],
            }
        )
    return rows


def _compute_item_expected_score_deltas(df: pd.DataFrame, spec: UltimatumGameSpec) -> pd.DataFrame:
    group_cols = ["model", "item_id", "emotion", "intensity"]

    rate = (
        df.groupby(group_cols, as_index=False)
        .agg(decision_score_mean=("decision_score", "mean"), n=("decision_score", "size"))
    )

    for beh in spec.behavior_to_score:
        tmp = (
            df.assign(val=(df["behavior"] == beh).astype(int))
            .groupby(group_cols, as_index=False)
            .agg(**{f"p_{beh}": ("val", "mean")})
        )
        rate = rate.merge(tmp, on=group_cols, how="left")

    p_cols = [f"p_{b}" for b in spec.behavior_to_score]
    neutral = (
        rate[rate["emotion"] == "neutral"]
        .set_index(["model", "item_id"])[["decision_score_mean", *p_cols]]
        .rename(columns={"decision_score_mean": "neutral_decision_score", **{c: f"neutral_{c}" for c in p_cols}})
    )

    rate = rate[rate["emotion"] != "neutral"].copy()
    rate = rate.join(neutral, on=["model", "item_id"], how="left")
    missing = rate["neutral_decision_score"].isna()
    if missing.any():
        rate = rate[~missing].copy()
    if rate.empty:
        raise ValueError("No rows with neutral baseline; cannot compute deltas")

    rate["delta_decision_score"] = rate["decision_score_mean"] - rate["neutral_decision_score"]
    min_delta = -float(max(spec.behavior_to_score.values()))
    max_delta = float(max(spec.behavior_to_score.values()))
    rate["delta_dir"] = pd.cut(
        rate["delta_decision_score"],
        bins=[min_delta - 0.01, -1e-9, 1e-9, max_delta + 0.01],
        labels=["decrease", "no_change", "increase"],
    )
    return rate


def _compute_item_max_delta_summary(rate: pd.DataFrame, spec: UltimatumGameSpec) -> pd.DataFrame:
    idx_cols = ["model", "item_id"]
    p_neutral_cols = [f"neutral_p_{b}" for b in spec.behavior_to_score]
    neutral = (
        rate.drop_duplicates(subset=idx_cols)[idx_cols + ["neutral_decision_score", *p_neutral_cols]].set_index(idx_cols)
    )

    max_inc = (
        rate.sort_values(["delta_decision_score", "intensity", "emotion"], ascending=[False, True, True])
        .groupby(idx_cols)
        .head(1)
        .rename(
            columns={
                "emotion": "max_inc_emotion",
                "intensity": "max_inc_intensity",
                "decision_score_mean": "max_inc_score",
                "delta_decision_score": "max_inc_delta",
            }
        )
    )
    max_dec = (
        rate.sort_values(["delta_decision_score", "intensity", "emotion"], ascending=[True, True, True])
        .groupby(idx_cols)
        .head(1)
        .rename(
            columns={
                "emotion": "max_dec_emotion",
                "intensity": "max_dec_intensity",
                "decision_score_mean": "max_dec_score",
                "delta_decision_score": "max_dec_delta",
            }
        )
    )

    out = (
        neutral.reset_index()
        .merge(
            max_inc[idx_cols + ["max_inc_emotion", "max_inc_intensity", "max_inc_score", "max_inc_delta"]],
            on=idx_cols,
            how="left",
        )
        .merge(
            max_dec[idx_cols + ["max_dec_emotion", "max_dec_intensity", "max_dec_score", "max_dec_delta"]],
            on=idx_cols,
            how="left",
        )
    )
    return out.sort_values(idx_cols, kind="mergesort").reset_index(drop=True)


def _compute_aggregates(rate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_by_ei = (
        rate.groupby(["model", "emotion", "intensity"], as_index=False)
        .agg(
            delta_mean=("delta_decision_score", "mean"),
            delta_median=("delta_decision_score", "median"),
            frac_inc=("delta_decision_score", lambda x: float((x > 0).mean())),
            frac_dec=("delta_decision_score", lambda x: float((x < 0).mean())),
            n_items=("item_id", "nunique"),
        )
        .sort_values(["model", "emotion", "intensity"])
    )

    agg_by_e = (
        rate.groupby(["model", "emotion"], as_index=False)
        .agg(
            delta_mean=("delta_decision_score", "mean"),
            delta_median=("delta_decision_score", "median"),
            frac_inc=("delta_decision_score", lambda x: float((x > 0).mean())),
            frac_dec=("delta_decision_score", lambda x: float((x < 0).mean())),
            n_rows=("delta_decision_score", "size"),
            n_items=("item_id", "nunique"),
        )
        .sort_values(["model", "delta_mean"], ascending=[True, False])
    )
    return agg_by_ei, agg_by_e


def _write_report_md(
    md_path: Path,
    rate: pd.DataFrame,
    agg_by_emotion: pd.DataFrame,
    item_summary: pd.DataFrame,
    spec: UltimatumGameSpec,
) -> None:
    def md_table(df: pd.DataFrame, max_rows: int) -> str:
        d = df.head(max_rows).copy()
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                if "frac_" in c:
                    d[c] = d[c].map(lambda x: f"{x:.3f}")
                elif "delta" in c:
                    d[c] = d[c].map(lambda x: f"{x:+.3f}")
                elif "score" in c:
                    d[c] = d[c].map(lambda x: f"{x:.3f}")
                elif c.startswith("p_") or c.startswith("neutral_p_"):
                    d[c] = d[c].map(lambda x: f"{x:.3f}")
        return d.to_markdown(index=False)

    prefix = spec.output_prefix
    lines: list[str] = []
    lines.append(f"# Ultimatum Game ({spec.title_role}) — Expected decision score delta vs neutral")
    lines.append("")
    lines.append(spec.decision_encoding_md)
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- `{prefix}_item_expected_score_delta_vs_neutral.csv`")
    lines.append(f"- `{prefix}_item_expected_score_max_delta_summary.csv`")
    lines.append(f"- `{prefix}_expected_score_delta_aggregate_by_emotion_intensity.csv`")
    lines.append(f"- `{prefix}_expected_score_delta_aggregate_by_emotion.csv`")
    lines.append("")
    lines.append("## Aggregate shifts (by emotion, averaged over intensities)")
    lines.append("")

    for model, sub in agg_by_emotion.groupby("model"):
        lines.append(f"### {model}")
        lines.append("")
        cols = ["emotion", "delta_mean", "delta_median", "frac_inc", "frac_dec", "n_items"]
        lines.append(md_table(sub[cols].sort_values("delta_mean", ascending=False), max_rows=10))
        lines.append("")

    inc = item_summary.sort_values(["max_inc_delta", "model", "item_id"], ascending=[False, True, True], kind="mergesort")
    dec = item_summary.sort_values(["max_dec_delta", "model", "item_id"], ascending=[True, True, True], kind="mergesort")

    lines.append("## Largest item-level shifts (max over emotions/intensities)")
    lines.append("")
    lines.append(f"### Top increases (toward {spec.up_label})")
    lines.append("")
    cols = [
        "model",
        "item_id",
        "neutral_decision_score",
        "max_inc_emotion",
        "max_inc_intensity",
        "max_inc_score",
        "max_inc_delta",
    ]
    lines.append(md_table(inc[cols], max_rows=15))
    lines.append("")
    lines.append(f"### Top decreases (toward {spec.down_label})")
    lines.append("")
    cols = [
        "model",
        "item_id",
        "neutral_decision_score",
        "max_dec_emotion",
        "max_dec_intensity",
        "max_dec_score",
        "max_dec_delta",
    ]
    lines.append(md_table(dec[cols], max_rows=15))
    lines.append("")

    lines.append("## Direction counts (all item×emotion×intensity)")
    lines.append("")
    vc = rate["delta_dir"].value_counts().rename_axis("delta_dir").reset_index(name="count")
    lines.append(vc.to_markdown(index=False))
    lines.append("")

    md_path.write_text("\n".join(lines).rstrip() + "\n")


def run_from_report(report_path: Path, out_dir: Path | None, spec: UltimatumGameSpec) -> AnalysisOutputs:
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"Expected object in {report_path}")

    if out_dir is None:
        out_dir = report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model, run_dir in _iter_run_dirs(report, spec=spec):
        rows.extend(_load_records(run_dir=run_dir, model=model, spec=spec))

    if not rows:
        raise ValueError(f"No {spec.benchmark_substring} runs found in {report_path}")

    df = pd.DataFrame(rows)
    missing = [c for c in ["model", "item_id", "emotion", "intensity", "behavior", "decision_score"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    rate = _compute_item_expected_score_deltas(df, spec=spec)
    item_summary = _compute_item_max_delta_summary(rate, spec=spec)
    agg_by_ei, agg_by_e = _compute_aggregates(rate)

    prefix = spec.output_prefix
    out_rate = out_dir / f"{prefix}_item_expected_score_delta_vs_neutral.csv"
    out_item = out_dir / f"{prefix}_item_expected_score_max_delta_summary.csv"
    out_agg = out_dir / f"{prefix}_expected_score_delta_aggregate_by_emotion_intensity.csv"
    out_agg_e = out_dir / f"{prefix}_expected_score_delta_aggregate_by_emotion.csv"
    out_md = out_dir / f"{prefix}_item_expected_score_delta_report.md"

    rate.to_csv(out_rate, index=False)
    item_summary.to_csv(out_item, index=False)
    agg_by_ei.to_csv(out_agg, index=False)
    agg_by_e.to_csv(out_agg_e, index=False)
    _write_report_md(md_path=out_md, rate=rate, agg_by_emotion=agg_by_e, item_summary=item_summary, spec=spec)

    return AnalysisOutputs(
        item_expected_score_deltas_path=str(out_rate),
        item_max_delta_summary_path=str(out_item),
        aggregate_by_emotion_intensity_path=str(out_agg),
        aggregate_by_emotion_path=str(out_agg_e),
        report_md_path=str(out_md),
        item_expected_score_deltas=rate,
        item_max_delta_summary=item_summary,
        aggregate_by_emotion_intensity=agg_by_ei,
        aggregate_by_emotion=agg_by_e,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--role",
        choices=["proposer", "responder"],
        default=None,
        help="Limit analysis to one role (default: run both proposer and responder)",
    )
    p.add_argument("--report", type=Path, required=True, help="Path to memory_experiment_series_*_report.json")
    p.add_argument("--out_dir", type=Path, default=None, help="Where to write outputs (default: report parent dir)")
    args = p.parse_args(argv)

    if args.role is None:
        any_role_ran = False
        for spec in (PROPOSER_SPEC, RESPONDER_SPEC):
            try:
                run_from_report(report_path=args.report, out_dir=args.out_dir, spec=spec)
                any_role_ran = True
            except ValueError as e:
                msg = str(e)
                if msg.startswith(f"No {spec.benchmark_substring} runs found in "):
                    continue
                raise
        if not any_role_ran:
            raise ValueError(f"No Ultimatum_Game_Proposer/Ultimatum_Game_Responder runs found in {args.report}")
        return 0

    spec = PROPOSER_SPEC if args.role == "proposer" else RESPONDER_SPEC
    run_from_report(report_path=args.report, out_dir=args.out_dir, spec=spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

