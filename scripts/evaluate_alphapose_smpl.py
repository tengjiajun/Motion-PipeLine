#!/usr/bin/env python
"""Evaluate AlphaPose SMPL raw npy files and emit reports for downstream review.

This script only evaluates the AlphaPose/source layer. It does not modify motion
data and does not make backend-specific assumptions about FRoM-W1 or ExBody.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_JOINTS_29 = {
    "root": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine": 3,
    "left_knee": 4,
    "right_knee": 5,
    "chest": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaPose SMPL raw npy quality and generate LLM text."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/smpl_raw"),
        help="Directory containing *_smpl_raw.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metrics/alphapose_quality"),
        help="Directory for JSON/Markdown/LLM reports.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS used for speed estimates.")
    parser.add_argument(
        "--low-kp-threshold",
        type=float,
        default=0.50,
        help="Frame is low confidence if mean keypoint score is below this.",
    )
    parser.add_argument(
        "--jump-factor",
        type=float,
        default=6.0,
        help="A frame-to-frame joint jump is abnormal if above median + factor * MAD.",
    )
    parser.add_argument(
        "--root-jump-threshold",
        type=float,
        default=0.35,
        help="Absolute root translation jump threshold in source units.",
    )
    parser.add_argument(
        "--pattern",
        default="*_smpl_raw.npy",
        help="Glob pattern under --input-dir.",
    )
    return parser.parse_args()


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


def load_frames(path: Path) -> list[dict[str, Any]]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if isinstance(obj, dict) and "frames" in obj:
        return list(obj["frames"])
    if isinstance(obj, np.ndarray):
        return list(obj)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported SMPL npy structure: {path}")


def motion_id_from_path(path: Path) -> str:
    name = path.stem
    suffix = "_smpl_raw"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def first_person(frame: dict[str, Any]) -> dict[str, Any] | None:
    results = frame.get("result") or []
    if len(results) == 0:
        return None
    return results[0]


def finite_mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def finite_min(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min())


def robust_jump_threshold(deltas: np.ndarray, factor: float) -> float:
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return math.inf
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if mad < 1e-8:
        return med * 3.0 + 1e-6
    return med + factor * 1.4826 * mad


def count_spikes(series: np.ndarray, factor: float, absolute_threshold: float | None = None) -> dict[str, Any]:
    if series.shape[0] < 2:
        return {"count": 0, "max_delta": None, "threshold": None, "frames": []}
    deltas = np.linalg.norm(np.diff(series, axis=0), axis=1)
    threshold = robust_jump_threshold(deltas, factor)
    if absolute_threshold is not None:
        threshold = max(threshold, absolute_threshold)
    frames = np.where(deltas > threshold)[0] + 1
    return {
        "count": int(frames.size),
        "max_delta": float(np.nanmax(deltas)) if deltas.size else None,
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "frames": [int(x) for x in frames[:20]],
    }


def choose_action(metrics: dict[str, Any]) -> str:
    if metrics["valid_frame_ratio"] < 0.90:
        return "reject_or_recapture"
    if metrics["multi_person_frame_count"] > 0:
        return "review_identity_tracking"
    if metrics["low_conf_frame_count"] > max(3, metrics["num_frames"] * 0.10):
        return "repair_then_visual_check"
    jump_count = sum(v["count"] for v in metrics["joint_jump_summary"].values())
    if jump_count > max(4, metrics["num_frames"] * 0.06):
        return "repair_then_visual_check"
    if metrics["root_jump"]["count"] > 0:
        return "repair_root_motion"
    return "usable"


def evaluate_file(path: Path, fps: float, low_kp_threshold: float, jump_factor: float, root_jump_threshold: float) -> dict[str, Any]:
    frames = load_frames(path)
    people_counts = []
    bbox_scores = []
    kp_means = []
    kp_mins = []
    xyz29 = []
    roots = []
    valid_indices = []

    for idx, frame in enumerate(frames):
        results = frame.get("result") or []
        people_counts.append(len(results))
        person = first_person(frame)
        if person is None:
            continue
        valid_indices.append(idx)
        bbox_scores.append(float(np.asarray(person.get("bbox_score", np.nan))))
        kp_score = np.asarray(person.get("kp_score", np.nan), dtype=np.float64)
        kp_means.append(float(np.nanmean(kp_score)))
        kp_mins.append(float(np.nanmin(kp_score)))
        if "pred_xyz_jts_29" in person:
            xyz29.append(np.asarray(person["pred_xyz_jts_29"], dtype=np.float64))
        if "transl" in person:
            roots.append(np.asarray(person["transl"], dtype=np.float64).reshape(3))
        elif "cam_root" in person:
            roots.append(np.asarray(person["cam_root"], dtype=np.float64).reshape(3))

    num_frames = len(frames)
    valid_frame_count = len(valid_indices)
    low_conf_frames = [valid_indices[i] for i, score in enumerate(kp_means) if score < low_kp_threshold]

    joint_jump_summary: dict[str, Any] = {}
    if len(xyz29) == valid_frame_count and valid_frame_count > 1:
        xyz_arr = np.stack(xyz29, axis=0)
        for name, joint_idx in DEFAULT_JOINTS_29.items():
            if joint_idx < xyz_arr.shape[1]:
                joint_jump_summary[name] = count_spikes(xyz_arr[:, joint_idx, :], jump_factor)

        upper_names = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        lower_names = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        upper_jump_count = int(sum(joint_jump_summary.get(k, {}).get("count", 0) for k in upper_names))
        lower_jump_count = int(sum(joint_jump_summary.get(k, {}).get("count", 0) for k in lower_names))
    else:
        upper_jump_count = 0
        lower_jump_count = 0

    root_jump = {"count": 0, "max_delta": None, "threshold": None, "frames": []}
    if len(roots) == valid_frame_count and valid_frame_count > 1:
        root_jump = count_spikes(np.stack(roots, axis=0), jump_factor, root_jump_threshold)

    metrics = {
        "motion_id": motion_id_from_path(path),
        "source_file": str(path),
        "fps": fps,
        "num_frames": num_frames,
        "valid_frame_count": valid_frame_count,
        "valid_frame_ratio": float(valid_frame_count / num_frames) if num_frames else 0.0,
        "empty_frame_count": int(sum(1 for c in people_counts if c == 0)),
        "multi_person_frame_count": int(sum(1 for c in people_counts if c > 1)),
        "people_counts": sorted(set(int(c) for c in people_counts)),
        "mean_bbox_score": finite_mean(bbox_scores),
        "mean_kp_score": finite_mean(kp_means),
        "min_kp_score": finite_min(kp_mins),
        "low_conf_frame_count": len(low_conf_frames),
        "low_conf_frames": [int(x) for x in low_conf_frames[:30]],
        "joint_jump_summary": joint_jump_summary,
        "upper_body_jump_count": upper_jump_count,
        "lower_body_jump_count": lower_jump_count,
        "root_jump": root_jump,
    }
    metrics["recommended_action"] = choose_action(metrics)
    return metrics


def llm_text(metrics: dict[str, Any]) -> str:
    abnormal_joints = [
        f"{name}:{summary['count']}"
        for name, summary in metrics["joint_jump_summary"].items()
        if summary.get("count", 0) > 0
    ]
    abnormal = ", ".join(abnormal_joints[:12]) if abnormal_joints else "none"
    return "\n".join(
        [
            f"Motion ID: {metrics['motion_id']}",
            "Layer: AlphaPose/SMPL source quality.",
            f"Frames: {metrics['valid_frame_count']}/{metrics['num_frames']} valid "
            f"({metrics['valid_frame_ratio']:.3f}).",
            f"Detection: empty_frames={metrics['empty_frame_count']}, "
            f"multi_person_frames={metrics['multi_person_frame_count']}, "
            f"people_counts={metrics['people_counts']}.",
            f"Confidence: mean_bbox_score={metrics['mean_bbox_score']:.4f}, "
            f"mean_kp_score={metrics['mean_kp_score']:.4f}, min_kp_score={metrics['min_kp_score']:.4f}, "
            f"low_conf_frame_count={metrics['low_conf_frame_count']}.",
            f"Temporal anomalies: upper_body_jump_count={metrics['upper_body_jump_count']}, "
            f"lower_body_jump_count={metrics['lower_body_jump_count']}, "
            f"root_jump_count={metrics['root_jump']['count']}, abnormal_joints={abnormal}.",
            f"Recommended source-layer action: {metrics['recommended_action']}.",
            "Instruction to motion-editing LLM: do not edit alphapose-results.json or smpl_raw.npy. "
            "Use this source-quality summary only as reliability evidence. If action is usable, continue "
            "to reference/canonical evaluation. If action asks for repair, run source repair/smoothing first "
            "and evaluate the repaired SMPL before generating backend-specific inputs.",
        ]
    )


def markdown_report(metrics: dict[str, Any], llm: str) -> str:
    joint_rows = []
    for name, summary in metrics["joint_jump_summary"].items():
        if summary.get("count", 0) > 0:
            joint_rows.append(
                f"| {name} | {summary['count']} | {summary['max_delta']:.6f} | {summary['threshold']:.6f} | {summary['frames']} |"
            )
    joint_table = "\n".join(joint_rows) if joint_rows else "| none | 0 | - | - | - |"
    return f"""# AlphaPose SMPL Quality Report: {metrics['motion_id']}

## Summary

| Metric | Value |
|---|---:|
| frames | {metrics['num_frames']} |
| valid frames | {metrics['valid_frame_count']} |
| valid frame ratio | {metrics['valid_frame_ratio']:.3f} |
| empty frames | {metrics['empty_frame_count']} |
| multi-person frames | {metrics['multi_person_frame_count']} |
| mean bbox score | {metrics['mean_bbox_score']:.4f} |
| mean keypoint score | {metrics['mean_kp_score']:.4f} |
| min keypoint score | {metrics['min_kp_score']:.4f} |
| low-confidence frames | {metrics['low_conf_frame_count']} |
| upper-body jump count | {metrics['upper_body_jump_count']} |
| lower-body jump count | {metrics['lower_body_jump_count']} |
| root jump count | {metrics['root_jump']['count']} |
| recommended action | {metrics['recommended_action']} |

## Abnormal Joint Jumps

| Joint | Count | Max delta | Threshold | Example frames |
|---|---:|---:|---:|---|
{joint_table}

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
        (output_dir / f"{motion_id}_quality.json").write_text(
            json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{motion_id}_llm.txt").write_text(llm + "\n", encoding="utf-8")
        (output_dir / f"{motion_id}_report.md").write_text(markdown_report(metrics, llm), encoding="utf-8")
        llm_blocks.append(llm)
        rows.append(
            {
                "motion_id": motion_id,
                "num_frames": metrics["num_frames"],
                "valid_frame_ratio": metrics["valid_frame_ratio"],
                "empty_frame_count": metrics["empty_frame_count"],
                "multi_person_frame_count": metrics["multi_person_frame_count"],
                "mean_bbox_score": metrics["mean_bbox_score"],
                "mean_kp_score": metrics["mean_kp_score"],
                "min_kp_score": metrics["min_kp_score"],
                "low_conf_frame_count": metrics["low_conf_frame_count"],
                "upper_body_jump_count": metrics["upper_body_jump_count"],
                "lower_body_jump_count": metrics["lower_body_jump_count"],
                "root_jump_count": metrics["root_jump"]["count"],
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
        "AlphaPose source-layer quality summaries for a motion-editing LLM.\n\n"
        + "\n\n---\n\n".join(llm_blocks)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob(args.pattern))
    paths = [p for p in paths if p.is_file() and p.stat().st_size > 0]
    if not paths:
        raise SystemExit(f"No files matched {args.input_dir / args.pattern}")

    metrics_list = [
        evaluate_file(p, args.fps, args.low_kp_threshold, args.jump_factor, args.root_jump_threshold)
        for p in paths
    ]
    write_outputs(metrics_list, args.output_dir)
    print(f"Wrote {len(metrics_list)} AlphaPose quality reports to {args.output_dir}")


if __name__ == "__main__":
    main()
