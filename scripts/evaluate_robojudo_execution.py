#!/usr/bin/env python
"""Evaluate RoboJuDo rollout execution quality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


UPPER_KEYWORDS = ("shoulder", "elbow", "torso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RoboJuDo rollout.npz files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/metrics/execution_robojudo"))
    parser.add_argument("--fall-height-threshold", type=float, default=0.75)
    parser.add_argument("--fall-tilt-deg-threshold", type=float, default=45.0)
    parser.add_argument("--tracking-rmse-warn", type=float, default=0.45)
    parser.add_argument("--tracking-rmse-fail", type=float, default=0.65)
    parser.add_argument("--base-drift-warn", type=float, default=0.35)
    parser.add_argument("--base-drift-fail", type=float, default=0.80)
    return parser.parse_args()


def quat_xyzw_to_rpy(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * np.pi / 2.0, np.arcsin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))


def evaluate_rollout(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in data["joint_names"]]
    dof = data["dof_pos"].astype(np.float32)
    ref = data["ref_dof_pos"].astype(np.float32)
    pd_target = data["pd_target"].astype(np.float32)
    base_pos = data["base_pos"].astype(np.float32)
    base_quat = data["base_quat"].astype(np.float32)
    dt = float(data["dt"][0])
    valid = np.isfinite(ref).all(axis=1)
    err = dof[valid] - ref[valid]
    pd_err = dof - pd_target

    upper = [idx for idx, name in enumerate(names) if any(key in name for key in UPPER_KEYWORDS)]
    lower = [idx for idx in range(len(names)) if idx not in upper]
    joint_rmse = np.sqrt(np.mean(np.square(err), axis=0))
    worst = np.argsort(joint_rmse)[-5:][::-1]
    rpy = quat_xyzw_to_rpy(base_quat)
    tilt = np.linalg.norm(rpy[:, :2], axis=1)
    dof_speed = np.diff(dof, axis=0) / max(dt, 1e-8)
    ref_speed = np.diff(ref, axis=0) / max(dt, 1e-8)

    metrics: dict[str, Any] = {
        "motion_id": path.parent.name,
        "rollout_file": str(path),
        "steps": int(dof.shape[0]),
        "duration_s": float(dof.shape[0] * dt),
        "tracking_rmse_all": rmse(err),
        "tracking_mae_all": float(np.mean(np.abs(err))),
        "tracking_max_abs_all": float(np.max(np.abs(err))),
        "tracking_rmse_upper": rmse(err[:, upper]) if upper else None,
        "tracking_rmse_lower": rmse(err[:, lower]) if lower else None,
        "pd_target_rmse_all": rmse(pd_err),
        "base_xy_drift_m": float(np.linalg.norm(base_pos[-1, :2] - base_pos[0, :2])),
        "base_xy_path_m": float(np.linalg.norm(np.diff(base_pos[:, :2], axis=0), axis=1).sum()),
        "base_z_mean_m": float(np.mean(base_pos[:, 2])),
        "base_z_min_m": float(np.min(base_pos[:, 2])),
        "base_z_range_m": float(np.ptp(base_pos[:, 2])),
        "base_tilt_mean_deg": float(np.degrees(np.mean(tilt))),
        "base_tilt_max_deg": float(np.degrees(np.max(tilt))),
        "dof_speed_max_rad_s": float(np.max(np.abs(dof_speed))),
        "ref_speed_max_rad_s": float(np.max(np.abs(ref_speed))),
        "worst_joint_rmse": [
            {"joint": names[idx], "rmse": float(joint_rmse[idx])}
            for idx in worst
        ],
        "fall_flag": bool(
            np.min(base_pos[:, 2]) < args.fall_height_threshold
            or np.max(tilt) > np.deg2rad(args.fall_tilt_deg_threshold)
        ),
    }
    metrics["recommended_action"] = recommend(metrics, args)
    return metrics


def recommend(metrics: dict[str, Any], args: argparse.Namespace) -> str:
    if metrics["fall_flag"]:
        return "execution_failed_fall_or_large_tilt"
    if metrics["base_xy_drift_m"] > args.base_drift_fail:
        return "execution_failed_large_drift"
    if metrics["tracking_rmse_all"] > args.tracking_rmse_fail:
        return "execution_failed_tracking"
    if metrics["tracking_rmse_all"] > args.tracking_rmse_warn:
        return "tracking_needs_improvement"
    if metrics["base_xy_drift_m"] > args.base_drift_warn:
        return "balance_or_footstep_review"
    return "execution_usable"


def llm_text(metrics: dict[str, Any]) -> str:
    worst = "; ".join(f"{x['joint']}={x['rmse']:.3f}" for x in metrics["worst_joint_rmse"])
    return "\n".join(
        [
            f"Motion ID: {metrics['motion_id']}",
            "Layer: execution/simulation quality from RoboJuDo rollout.",
            f"Duration: {metrics['duration_s']:.2f}s, steps={metrics['steps']}.",
            f"Tracking: rmse_all={metrics['tracking_rmse_all']:.3f} rad, "
            f"rmse_upper={metrics['tracking_rmse_upper']:.3f} rad, "
            f"rmse_lower={metrics['tracking_rmse_lower']:.3f} rad, "
            f"max_abs={metrics['tracking_max_abs_all']:.3f} rad.",
            f"Base stability: xy_drift={metrics['base_xy_drift_m']:.3f} m, "
            f"z_min={metrics['base_z_min_m']:.3f} m, z_range={metrics['base_z_range_m']:.3f} m, "
            f"tilt_max={metrics['base_tilt_max_deg']:.1f} deg, fall_flag={metrics['fall_flag']}.",
            f"Worst tracking joints: {worst}.",
            f"Recommended execution action: {metrics['recommended_action']}.",
            "Instruction to motion-editing LLM: use this only after canonical and backend conversion pass. "
            "If fall_flag or large drift appears, prefer slower/smaller/stabler reference edits rather than changing AlphaPose raw data.",
        ]
    )


def markdown_report(metrics: dict[str, Any], llm: str) -> str:
    worst_rows = "\n".join(
        f"| {x['joint']} | {x['rmse']:.4f} |" for x in metrics["worst_joint_rmse"]
    )
    return f"""# RoboJuDo Execution Report: {metrics['motion_id']}

## Summary

| Metric | Value |
|---|---:|
| duration sec | {metrics['duration_s']:.3f} |
| tracking rmse all rad | {metrics['tracking_rmse_all']:.4f} |
| tracking rmse upper rad | {metrics['tracking_rmse_upper']:.4f} |
| tracking rmse lower rad | {metrics['tracking_rmse_lower']:.4f} |
| tracking max abs rad | {metrics['tracking_max_abs_all']:.4f} |
| base xy drift m | {metrics['base_xy_drift_m']:.4f} |
| base z min m | {metrics['base_z_min_m']:.4f} |
| base z range m | {metrics['base_z_range_m']:.4f} |
| base tilt max deg | {metrics['base_tilt_max_deg']:.3f} |
| fall flag | {metrics['fall_flag']} |
| recommended action | {metrics['recommended_action']} |

## Worst Tracking Joints

| Joint | RMSE rad |
|---|---:|
{worst_rows}

## LLM Text

```text
{llm}
```
"""


def write_outputs(metrics_list: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    llm_blocks = []
    for metrics in metrics_list:
        motion_id = metrics["motion_id"]
        llm = llm_text(metrics)
        llm_blocks.append(llm)
        (output_dir / f"{motion_id}_execution_quality.json").write_text(
            json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{motion_id}_execution_llm.txt").write_text(llm + "\n", encoding="utf-8")
        (output_dir / f"{motion_id}_execution_report.md").write_text(markdown_report(metrics, llm), encoding="utf-8")
        rows.append(
            {
                "motion_id": motion_id,
                "duration_s": metrics["duration_s"],
                "tracking_rmse_all": metrics["tracking_rmse_all"],
                "tracking_rmse_upper": metrics["tracking_rmse_upper"],
                "tracking_rmse_lower": metrics["tracking_rmse_lower"],
                "tracking_max_abs_all": metrics["tracking_max_abs_all"],
                "pd_target_rmse_all": metrics["pd_target_rmse_all"],
                "base_xy_drift_m": metrics["base_xy_drift_m"],
                "base_z_min_m": metrics["base_z_min_m"],
                "base_z_range_m": metrics["base_z_range_m"],
                "base_tilt_max_deg": metrics["base_tilt_max_deg"],
                "fall_flag": metrics["fall_flag"],
                "recommended_action": metrics["recommended_action"],
            }
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(to_jsonable(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "llm_batch_prompt.txt").write_text(
        "RoboJuDo execution quality summaries for a motion-editing LLM.\n\n"
        + "\n\n---\n\n".join(llm_blocks)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rollout_paths = sorted(args.input_dir.glob("*/rollout.npz"))
    if not rollout_paths:
        raise FileNotFoundError(f"No rollout.npz files found under {args.input_dir}")
    metrics_list = [evaluate_rollout(path, args) for path in rollout_paths]
    write_outputs(metrics_list, args.output_dir)
    print(f"Wrote {len(metrics_list)} execution reports to {args.output_dir}")


if __name__ == "__main__":
    main()
