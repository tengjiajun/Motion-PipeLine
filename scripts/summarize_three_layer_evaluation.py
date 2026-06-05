#!/usr/bin/env python
"""Summarize AlphaPose, canonical, and execution evaluations for LLM analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize three-layer motion evaluation results.")
    parser.add_argument(
        "--alphapose-summary",
        type=Path,
        default=Path("data/metrics/alphapose_repaired_quality/summary.csv"),
        help="AlphaPose/source-layer summary.csv, preferably after SMPL repair.",
    )
    parser.add_argument(
        "--canonical-summary",
        type=Path,
        default=Path("data/metrics/canonical_edited_quality/summary.csv"),
        help="Canonical/reference-layer summary.csv.",
    )
    parser.add_argument(
        "--execution-summary",
        type=Path,
        default=Path("data/metrics/execution_robojudo_fromw1_pkl_canonical_v2/summary.csv"),
        help="RoboJuDo execution-layer summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metrics/three_layer_summary_fromw1_v2"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm_id(value: str) -> str:
    out = value
    suffixes = [
        "_smpl_repaired_compact",
        "_smpl_repaired",
        "_smpl_raw",
        "_canonical_motion_v2",
        "_canonical_motion",
        "_623",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", "None", "nan", "NaN"}:
        return default
    return float(value)


def s(row: dict[str, str], key: str, default: str = "") -> str:
    return row.get(key, default) or default


def index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {norm_id(row["motion_id"]): row for row in rows}


def source_status(row: dict[str, str] | None) -> str:
    if row is None:
        return "missing"
    return s(row, "recommended_action", "unknown")


def canonical_status(row: dict[str, str] | None) -> str:
    if row is None:
        return "missing"
    return s(row, "recommended_action", "unknown")


def execution_status(row: dict[str, str] | None) -> str:
    if row is None:
        return "missing"
    return s(row, "recommended_action", "unknown")


def diagnosis(alpha: dict[str, str] | None, canonical: dict[str, str] | None, execution: dict[str, str] | None) -> tuple[str, list[str]]:
    notes: list[str] = []
    a_status = source_status(alpha)
    c_status = canonical_status(canonical)
    e_status = execution_status(execution)

    if a_status not in {"usable", "missing"}:
        notes.append("source layer still has quality issues; avoid changing semantics before source repair is reviewed")
    if c_status == "smooth_reference":
        notes.append("canonical reference remains too sharp or unstable; prefer generic smoothing/stabilization edits")
    if c_status == "stabilize_lower_body":
        notes.append("canonical lower-body/root motion should be stabilized before backend execution")
    if e_status == "execution_failed_fall_or_large_tilt":
        notes.append("execution failed by fall or large base tilt; prioritize slower/smaller/stabler reference edits")
    elif e_status == "tracking_needs_improvement":
        notes.append("execution is stable enough but tracking error is high; reduce high-speed changes and difficult joints")
    elif e_status == "execution_usable":
        notes.append("execution is usable; semantic edits should only follow user feedback")

    if e_status.startswith("execution_failed"):
        overall = "execution_failed"
    elif a_status not in {"usable", "missing"}:
        overall = "source_review_needed"
    elif c_status not in {"usable", "missing"}:
        overall = "reference_edit_needed"
    elif e_status == "tracking_needs_improvement":
        overall = "execution_tracking_review"
    elif e_status == "execution_usable":
        overall = "usable_pending_user_review"
    else:
        overall = "review_needed"
    return overall, notes


def build_record(motion_id: str, alpha: dict[str, str] | None, canonical: dict[str, str] | None, execution: dict[str, str] | None) -> dict[str, Any]:
    overall, notes = diagnosis(alpha, canonical, execution)
    return {
        "motion_id": motion_id,
        "overall_status": overall,
        "source_layer": {
            "status": source_status(alpha),
            "valid_frame_ratio": f(alpha, "valid_frame_ratio") if alpha else None,
            "mean_kp_score": f(alpha, "mean_kp_score") if alpha else None,
            "upper_body_jump_count": f(alpha, "upper_body_jump_count") if alpha else None,
            "lower_body_jump_count": f(alpha, "lower_body_jump_count") if alpha else None,
            "root_jump_count": f(alpha, "root_jump_count") if alpha else None,
        },
        "canonical_layer": {
            "status": canonical_status(canonical),
            "max_keypoint_speed_mps": f(canonical, "max_keypoint_speed_mps") if canonical else None,
            "max_keypoint_accel_mps2": f(canonical, "max_keypoint_accel_mps2") if canonical else None,
            "max_keypoint_jerk_mps3": f(canonical, "max_keypoint_jerk_mps3") if canonical else None,
            "lower_body_motion_ratio": f(canonical, "lower_body_motion_ratio") if canonical else None,
            "violation_count": f(canonical, "violation_count") if canonical else None,
        },
        "execution_layer": {
            "status": execution_status(execution),
            "tracking_rmse_all": f(execution, "tracking_rmse_all") if execution else None,
            "tracking_rmse_upper": f(execution, "tracking_rmse_upper") if execution else None,
            "tracking_rmse_lower": f(execution, "tracking_rmse_lower") if execution else None,
            "base_xy_drift_m": f(execution, "base_xy_drift_m") if execution else None,
            "base_z_min_m": f(execution, "base_z_min_m") if execution else None,
            "base_tilt_max_deg": f(execution, "base_tilt_max_deg") if execution else None,
            "fall_flag": s(execution, "fall_flag") if execution else None,
        },
        "diagnosis_notes": notes,
    }


def llm_motion_block(record: dict[str, Any]) -> str:
    src = record["source_layer"]
    can = record["canonical_layer"]
    exe = record["execution_layer"]
    notes = "\n".join(f"- {note}" for note in record["diagnosis_notes"])
    return f"""Motion ID: {record['motion_id']}
Overall status: {record['overall_status']}

Source layer:
- status={src['status']}
- valid_frame_ratio={src['valid_frame_ratio']}
- mean_kp_score={src['mean_kp_score']}
- upper_body_jump_count={src['upper_body_jump_count']}
- lower_body_jump_count={src['lower_body_jump_count']}
- root_jump_count={src['root_jump_count']}

Canonical/reference layer:
- status={can['status']}
- max_speed={can['max_keypoint_speed_mps']} m/s
- max_accel={can['max_keypoint_accel_mps2']} m/s^2
- max_jerk={can['max_keypoint_jerk_mps3']} m/s^3
- lower_body_motion_ratio={can['lower_body_motion_ratio']}
- violation_count={can['violation_count']}

Execution/RoboJuDo layer:
- status={exe['status']}
- tracking_rmse_all={exe['tracking_rmse_all']} rad
- tracking_rmse_upper={exe['tracking_rmse_upper']} rad
- tracking_rmse_lower={exe['tracking_rmse_lower']} rad
- base_xy_drift={exe['base_xy_drift_m']} m
- base_z_min={exe['base_z_min_m']} m
- base_tilt_max={exe['base_tilt_max_deg']} deg
- fall_flag={exe['fall_flag']}

Diagnosis notes:
{notes if notes else "- none"}
"""


def build_llm_prompt(records: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    blocks = "\n---\n".join(llm_motion_block(record) for record in records)
    return f"""You are analyzing a humanoid motion imitation pipeline with three evaluation layers.

Layer meanings:
1. Source layer checks AlphaPose/SMPL data quality. If this fails, do not edit semantics; repair or recapture source data first.
2. Canonical/reference layer checks whether the reference motion is smooth, stable, and suitable for retargeting. Generic edits should only change smoothness, lower-body stability, root/torso offset, or speed unless user feedback provides semantic intent.
3. Execution/RoboJuDo layer checks whether H1 actually followed the reference in simulation. Use this layer to decide whether to slow down, reduce amplitude, stabilize lower body, or keep the motion for user review.

Aggregate:
{json.dumps(aggregate, ensure_ascii=False, indent=2)}

Required analysis output:
- Identify motions that should not be semantically edited yet.
- Identify motions that need only generic reference repair.
- Identify motions that failed execution and require priority edits.
- For each motion, propose structured edit operations only when justified by these metrics.
- Do not infer action semantics from filenames.

Per-motion summaries:
{blocks}
"""


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "num_motions": len(records),
        "overall_status_counts": dict(Counter(record["overall_status"] for record in records)),
        "source_status_counts": dict(Counter(record["source_layer"]["status"] for record in records)),
        "canonical_status_counts": dict(Counter(record["canonical_layer"]["status"] for record in records)),
        "execution_status_counts": dict(Counter(record["execution_layer"]["status"] for record in records)),
    }
    (output_dir / "three_layer_summary.json").write_text(
        json.dumps({"aggregate": aggregate, "motions": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "three_layer_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "motion_id",
            "overall_status",
            "source_status",
            "canonical_status",
            "execution_status",
            "tracking_rmse_all",
            "base_xy_drift_m",
            "base_tilt_max_deg",
            "canonical_lower_body_motion_ratio",
            "canonical_violation_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "motion_id": record["motion_id"],
                    "overall_status": record["overall_status"],
                    "source_status": record["source_layer"]["status"],
                    "canonical_status": record["canonical_layer"]["status"],
                    "execution_status": record["execution_layer"]["status"],
                    "tracking_rmse_all": record["execution_layer"]["tracking_rmse_all"],
                    "base_xy_drift_m": record["execution_layer"]["base_xy_drift_m"],
                    "base_tilt_max_deg": record["execution_layer"]["base_tilt_max_deg"],
                    "canonical_lower_body_motion_ratio": record["canonical_layer"]["lower_body_motion_ratio"],
                    "canonical_violation_count": record["canonical_layer"]["violation_count"],
                }
            )
    prompt = build_llm_prompt(records, aggregate)
    (output_dir / "llm_three_layer_analysis_prompt.txt").write_text(prompt, encoding="utf-8")
    report_rows = "\n".join(
        f"| {r['motion_id']} | {r['overall_status']} | {r['source_layer']['status']} | "
        f"{r['canonical_layer']['status']} | {r['execution_layer']['status']} |"
        for r in records
    )
    (output_dir / "three_layer_report.md").write_text(
        f"""# Three-Layer Evaluation Summary

## Aggregate

```json
{json.dumps(aggregate, ensure_ascii=False, indent=2)}
```

## Motions

| Motion | Overall | Source | Canonical | Execution |
|---|---|---|---|---|
{report_rows}
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    alpha = index_rows(read_csv(args.alphapose_summary))
    canonical = index_rows(read_csv(args.canonical_summary))
    execution = index_rows(read_csv(args.execution_summary))
    ids = sorted(set(alpha) | set(canonical) | set(execution))
    records = [build_record(motion_id, alpha.get(motion_id), canonical.get(motion_id), execution.get(motion_id)) for motion_id in ids]
    write_outputs(records, args.output_dir)
    print(f"Wrote three-layer summary for {len(records)} motions to {args.output_dir}")


if __name__ == "__main__":
    main()
