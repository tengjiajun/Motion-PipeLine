#!/usr/bin/env python
"""Evaluate canonical_motion.npz reference quality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


JOINT = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
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
    parser = argparse.ArgumentParser(description="Evaluate canonical motion reference quality.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/canonical"))
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--glob", default="*_canonical_motion.npz")
    parser.add_argument("--rules", type=Path, default=Path("configs/canonical_eval_rules_h1_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/metrics/canonical_quality"))
    parser.add_argument(
        "--action-label",
        default="",
        help="Optional semantic action label. Default empty means generic quality evaluation only.",
    )
    parser.add_argument(
        "--infer-action-from-name",
        action="store_true",
        help="Experimental fallback: infer action type from motion_id keywords.",
    )
    return parser.parse_args()


def load_rules(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())


def range_xyz(points: np.ndarray) -> np.ndarray:
    return (points.max(axis=0) - points.min(axis=0)).astype(np.float32)


def max_norm_diff(points: np.ndarray, fps: float, order: int) -> float:
    arr = points.astype(np.float64)
    for _ in range(order):
        if arr.shape[0] < 2:
            return 0.0
        arr = np.diff(arr, axis=0) * fps
    return float(np.linalg.norm(arr.reshape(arr.shape[0], -1, 3), axis=-1).max())


def count_spikes(points: np.ndarray, fps: float, warn_speed: float, warn_accel: float) -> int:
    speed = np.linalg.norm(np.diff(points, axis=0) * fps, axis=-1) if points.shape[0] > 1 else np.zeros((0, points.shape[1]))
    accel = (
        np.linalg.norm(np.diff(points, n=2, axis=0) * fps * fps, axis=-1)
        if points.shape[0] > 2
        else np.zeros((0, points.shape[1]))
    )
    return int(np.count_nonzero(speed > warn_speed) + np.count_nonzero(accel > warn_accel))


def zero_crossing_count(series: np.ndarray) -> int:
    centered = series - np.mean(series)
    if centered.size < 3 or np.max(np.abs(centered)) < 1e-8:
        return 0
    signs = np.sign(centered)
    signs[signs == 0] = 1
    return int(np.count_nonzero(signs[1:] != signs[:-1]) // 2)


def above_ratio(hand: np.ndarray, shoulder: np.ndarray, margin: float = 0.0) -> float:
    return float(np.mean(hand[:, 1] > shoulder[:, 1] + margin))


def infer_action(motion_id: str, rules: dict[str, Any]) -> str | None:
    keys = sorted(rules["semantic_action_rules"].keys(), key=len, reverse=True)
    for key in keys:
        if key in motion_id:
            return key
    return None


def compute_metrics(path: Path, rules: dict[str, Any], action_label: str = "", infer_action_from_name: bool = False) -> dict[str, Any]:
    data = np.load(str(path), allow_pickle=False)
    motion_id = str(data["motion_id"])
    fps = float(data["fps"])
    joints = data["motionx52_joints"].astype(np.float32)
    smpl = data["smpl24_joints"].astype(np.float32)
    source_root = data["root_translation"].astype(np.float32)
    num_frames = int(data["num_frames"])
    duration = num_frames / fps if fps > 0 else 0.0

    kin = rules["simple_kinematic_constraints"]
    smooth_rules = kin["smoothness"]
    root_rules = kin["root_motion"]
    lower_rules = kin["lower_body_stability"]

    pelvis = joints[:, JOINT["pelvis"]]
    left_hand = joints[:, JOINT["left_wrist"]]
    right_hand = joints[:, JOINT["right_wrist"]]
    left_shoulder = joints[:, JOINT["left_shoulder"]]
    right_shoulder = joints[:, JOINT["right_shoulder"]]
    left_ankle = joints[:, JOINT["left_ankle"]]
    right_ankle = joints[:, JOINT["right_ankle"]]
    left_knee = joints[:, JOINT["left_knee"]]
    right_knee = joints[:, JOINT["right_knee"]]
    head = joints[:, JOINT["head"]]
    spine3 = joints[:, JOINT["spine3"]]

    upper_ids = [JOINT[x] for x in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]]
    lower_ids = [JOINT[x] for x in ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"]]
    upper_path = sum(path_length(joints[:, i]) for i in upper_ids)
    lower_path = sum(path_length(joints[:, i]) for i in lower_ids)
    lower_body_motion_ratio = float(lower_path / max(upper_path + lower_path, 1e-8))

    root = pelvis
    root_xy = root[:, [0, 2]]
    root_range = range_xyz(root)
    left_wrist_range = range_xyz(left_hand)
    right_wrist_range = range_xyz(right_hand)
    ankle_xy_motion = max(path_length(left_ankle[:, [0, 2]]), path_length(right_ankle[:, [0, 2]]))
    knee_xy_motion = max(path_length(left_knee[:, [0, 2]]), path_length(right_knee[:, [0, 2]]))
    foot_height_change = max(float(np.ptp(joints[:, JOINT["left_foot"], 1])), float(np.ptp(joints[:, JOINT["right_foot"], 1])))
    pelvis_ankle_center_offset = np.linalg.norm((pelvis[:, [0, 2]] - 0.5 * (left_ankle[:, [0, 2]] + right_ankle[:, [0, 2]])), axis=-1)

    action_type = action_label.strip() or (infer_action(motion_id, rules) if infer_action_from_name else None)

    metrics: dict[str, Any] = {
        "motion_id": motion_id,
        "source_file": str(path),
        "fps": int(fps),
        "num_frames": num_frames,
        "duration_sec": duration,
        "max_keypoint_speed_mps": max_norm_diff(joints[:, :22], fps, order=1),
        "max_keypoint_accel_mps2": max_norm_diff(joints[:, :22], fps, order=2),
        "max_keypoint_jerk_mps3": max_norm_diff(joints[:, :22], fps, order=3),
        "spike_count": count_spikes(
            joints[:, :22],
            fps,
            smooth_rules["warn_keypoint_speed_mps"],
            smooth_rules["warn_keypoint_accel_mps2"],
        ),
        "source_root_path_length_m": path_length(source_root),
        "root_path_length_m": path_length(root),
        "root_lateral_drift_m": float(np.ptp(root_xy[:, 0])) if root_xy.size else 0.0,
        "root_forward_backward_drift_m": float(np.ptp(root_xy[:, 1])) if root_xy.size else 0.0,
        "root_height_range_m": float(np.ptp(root[:, 1])) if root.size else 0.0,
        "root_max_speed_mps": max_norm_diff(root[:, None, :], fps, order=1),
        "left_hand_path_m": path_length(left_hand),
        "right_hand_path_m": path_length(right_hand),
        "left_wrist_amplitude_xyz_m": left_wrist_range,
        "right_wrist_amplitude_xyz_m": right_wrist_range,
        "left_hand_above_shoulder_ratio": above_ratio(left_hand, left_shoulder),
        "right_hand_above_shoulder_ratio": above_ratio(right_hand, right_shoulder),
        "lower_body_motion_ratio": lower_body_motion_ratio,
        "foot_height_change_m": foot_height_change,
        "ankle_xy_motion_m": ankle_xy_motion,
        "knee_xy_motion_m": knee_xy_motion,
        "max_pelvis_to_ankle_center_xy_offset_m": float(np.max(pelvis_ankle_center_offset)),
        "head_height_drop_m": float(np.max(head[:, 1]) - np.min(head[:, 1])),
        "torso_path_m": path_length(spine3),
        "left_right_hand_amplitude_ratio": float(
            np.linalg.norm(left_wrist_range) / max(np.linalg.norm(right_wrist_range), 1e-8)
        ),
        "right_wrist_repetition_count": zero_crossing_count(right_hand[:, 0]),
        "left_wrist_repetition_count": zero_crossing_count(left_hand[:, 0]),
        "action_type": action_type,
        "violations": [],
        "suggested_edit_ops": [],
    }

    add_violations(metrics, rules)
    if action_type:
        evaluate_semantics(metrics, rules, left_hand, right_hand, left_shoulder, right_shoulder, head, spine3, smpl)
    metrics["recommended_action"] = recommend(metrics)
    return metrics


def add_violation(
    metrics: dict[str, Any],
    name: str,
    value: float,
    threshold: float,
    severity: str,
    edit: str | None = None,
    relation: str = "above",
) -> None:
    metrics["violations"].append(
        {
            "name": name,
            "value": float(value),
            "threshold": float(threshold),
            "severity": severity,
            "relation": relation,
        }
    )
    if edit and edit not in metrics["suggested_edit_ops"]:
        metrics["suggested_edit_ops"].append(edit)


def add_violations(metrics: dict[str, Any], rules: dict[str, Any]) -> None:
    kin = rules["simple_kinematic_constraints"]
    checks = [
        ("max_keypoint_speed_mps", kin["smoothness"]["warn_keypoint_speed_mps"], kin["smoothness"]["max_keypoint_speed_mps"], "smooth_segment target=high_speed_joints"),
        ("max_keypoint_accel_mps2", kin["smoothness"]["warn_keypoint_accel_mps2"], kin["smoothness"]["max_keypoint_accel_mps2"], "smooth_segment target=high_acceleration_joints"),
        ("max_keypoint_jerk_mps3", kin["smoothness"]["warn_keypoint_jerk_mps3"], kin["smoothness"]["max_keypoint_jerk_mps3"], "smooth_segment target=high_jerk_joints"),
        ("root_lateral_drift_m", kin["root_motion"]["warn_root_lateral_drift_m"], kin["root_motion"]["max_root_lateral_drift_m"], "limit_root_translation"),
        ("root_forward_backward_drift_m", kin["root_motion"]["warn_root_forward_backward_drift_m"], kin["root_motion"]["max_root_forward_backward_drift_m"], "limit_root_translation"),
        ("root_height_range_m", kin["root_motion"]["warn_root_height_range_m"], kin["root_motion"]["max_root_height_range_m"], "smooth_root_height"),
        ("root_max_speed_mps", kin["root_motion"]["warn_root_speed_mps"], kin["root_motion"]["max_root_speed_mps"], "slow_down_segment target=root"),
        ("foot_height_change_m", kin["lower_body_stability"]["warn_foot_height_change_m"], kin["lower_body_stability"]["max_foot_height_change_m"], "stabilize_lower_body"),
        ("ankle_xy_motion_m", kin["lower_body_stability"]["warn_ankle_xy_motion_m"], kin["lower_body_stability"]["max_ankle_xy_motion_m"], "stabilize_lower_body"),
        ("knee_xy_motion_m", kin["lower_body_stability"]["warn_knee_xy_motion_m"], kin["lower_body_stability"]["max_knee_xy_motion_m"], "stabilize_lower_body"),
        ("lower_body_motion_ratio", kin["lower_body_stability"]["lower_body_motion_ratio_warn"], kin["lower_body_stability"]["lower_body_motion_ratio_fail"], "stabilize_lower_body"),
        ("max_pelvis_to_ankle_center_xy_offset_m", kin["balance_proxy"]["warn_pelvis_to_ankle_center_xy_offset_m"], kin["balance_proxy"]["max_pelvis_to_ankle_center_xy_offset_m"], "reduce_torso_or_root_offset"),
    ]
    for key, warn, fail, edit in checks:
        value = float(metrics[key])
        if value > fail:
            add_violation(metrics, key, value, fail, "fail", edit, relation="above")
        elif value > warn:
            add_violation(metrics, key, value, warn, "warn", edit, relation="above")


def evaluate_semantics(
    metrics: dict[str, Any],
    rules: dict[str, Any],
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    head: np.ndarray,
    spine3: np.ndarray,
    smpl: np.ndarray,
) -> None:
    action = metrics["action_type"]
    if action is None:
        return
    rule = rules["semantic_action_rules"][action]
    req = rule.get("required_features", {})
    avoid = rule.get("avoid", {})
    edits = rule.get("suggested_edit_if_failed", [])

    def fail_min(name: str, value: float) -> None:
        if name in req and value < float(req[name]):
            add_violation(
                metrics,
                name,
                value,
                float(req[name]),
                "semantic_fail",
                edits[0] if edits else None,
                relation="below",
            )

    def fail_max(name: str, value: float) -> None:
        if name in req and value > float(req[name]):
            add_violation(metrics, name, value, float(req[name]), "semantic_fail", edits[0] if edits else None, relation="above")
        if name in avoid and value > float(avoid[name]):
            add_violation(metrics, name, value, float(avoid[name]), "semantic_warn", edits[-1] if edits else None, relation="above")

    right_lateral_amp = float(metrics["right_wrist_amplitude_xyz_m"][0])
    left_lateral_amp = float(metrics["left_wrist_amplitude_xyz_m"][0])
    both_opening = float(np.ptp(np.linalg.norm(left_hand - right_hand, axis=-1)))
    left_arm_extension = float(np.max(np.linalg.norm(left_hand - left_shoulder, axis=-1)))
    right_arm_extension = float(np.max(np.linalg.norm(right_hand - right_shoulder, axis=-1)))
    torso_forward_motion = float(np.ptp(spine3[:, 2]))
    root_drift = max(float(metrics["root_lateral_drift_m"]), float(metrics["root_forward_backward_drift_m"]))

    fail_min("right_hand_above_shoulder_ratio_min", metrics["right_hand_above_shoulder_ratio"])
    fail_min("left_hand_above_shoulder_ratio_min", metrics["left_hand_above_shoulder_ratio"])
    fail_min("right_wrist_lateral_amplitude_min_m", right_lateral_amp)
    fail_max("right_wrist_lateral_amplitude_max_m", right_lateral_amp)
    fail_min("wrist_lateral_amplitude_min_m", min(left_lateral_amp, right_lateral_amp))
    fail_max("wrist_lateral_amplitude_max_m", max(left_lateral_amp, right_lateral_amp))
    fail_min("left_arm_extension_min_m", left_arm_extension)
    fail_min("right_arm_extension_min_m", right_arm_extension)
    fail_min("left_hand_lateral_displacement_min_m", left_lateral_amp)
    fail_min("right_hand_lateral_displacement_min_m", right_lateral_amp)
    fail_min("head_height_drop_min_m", metrics["head_height_drop_m"])
    fail_min("head_vertical_motion_min_m", metrics["head_height_drop_m"])
    fail_max("head_vertical_motion_max_m", metrics["head_height_drop_m"])
    fail_min("torso_forward_displacement_min_m", torso_forward_motion)
    fail_min("both_hands_opening_amplitude_min_m", both_opening)
    fail_min("both_hands_forward_or_lateral_motion_min_m", max(left_lateral_amp, right_lateral_amp, both_opening))
    fail_min("right_hand_peak_above_shoulder_min_m", float(np.max(right_hand[:, 1] - right_shoulder[:, 1])))
    fail_min("right_wrist_path_min_m", metrics["right_hand_path_m"])

    if "root_drift_max_m" in avoid and root_drift > float(avoid["root_drift_max_m"]):
        add_violation(metrics, "root_drift_max_m", root_drift, float(avoid["root_drift_max_m"]), "semantic_warn", "limit_root_translation", relation="above")
    if "lower_body_motion_ratio_max" in avoid and metrics["lower_body_motion_ratio"] > float(avoid["lower_body_motion_ratio_max"]):
        add_violation(metrics, "lower_body_motion_ratio_max", metrics["lower_body_motion_ratio"], float(avoid["lower_body_motion_ratio_max"]), "semantic_warn", "stabilize_lower_body", relation="above")
    if "left_hand_path_max_m" in avoid and metrics["left_hand_path_m"] > float(avoid["left_hand_path_max_m"]):
        add_violation(metrics, "left_hand_path_max_m", metrics["left_hand_path_m"], float(avoid["left_hand_path_max_m"]), "semantic_warn", "reduce_motion target=left_hand", relation="above")
    if "right_hand_path_max_m" in avoid and metrics["right_hand_path_m"] > float(avoid["right_hand_path_max_m"]):
        add_violation(metrics, "right_hand_path_max_m", metrics["right_hand_path_m"], float(avoid["right_hand_path_max_m"]), "semantic_warn", "reduce_motion target=right_hand", relation="above")


def recommend(metrics: dict[str, Any]) -> str:
    severities = {v["severity"] for v in metrics["violations"]}
    edits = set(metrics["suggested_edit_ops"])
    if "fail" in severities and any("smooth" in e for e in edits):
        return "smooth_reference"
    if "fail" in severities and any("stabilize_lower_body" in e for e in edits):
        return "stabilize_lower_body"
    if "semantic_fail" in severities and any(("increase" in e or "raise" in e or "extend" in e) for e in edits):
        return "increase_semantic_amplitude"
    if any(("reduce" in e or "limit" in e) for e in edits):
        return "reduce_amplitude"
    if metrics["violations"]:
        return "retarget_check_required"
    return "usable"


def llm_text(metrics: dict[str, Any]) -> str:
    top = metrics["violations"][:8]
    def describe_violation(v: dict[str, Any]) -> str:
        verb = "is below required" if v.get("relation") == "below" else "exceeds"
        return f"{v['name']}={v['value']:.4f} {verb} {v['threshold']:.4f} ({v['severity']})"

    problems = "; ".join(describe_violation(v) for v in top) or "none"
    edits = "; ".join(metrics["suggested_edit_ops"][:8]) or "none"
    return "\n".join(
        [
            f"Motion ID: {metrics['motion_id']}",
            "Layer: canonical/reference quality.",
            f"Action type: {metrics['action_type'] or 'unknown'}",
            f"Duration: {metrics['duration_sec']:.2f}s, frames={metrics['num_frames']}, fps={metrics['fps']}.",
            f"Smoothness: max_speed={metrics['max_keypoint_speed_mps']:.3f} m/s, "
            f"max_accel={metrics['max_keypoint_accel_mps2']:.3f} m/s^2, "
            f"max_jerk={metrics['max_keypoint_jerk_mps3']:.3f} m/s^3, spike_count={metrics['spike_count']}.",
            f"Stability: root_drift_xy=({metrics['root_lateral_drift_m']:.3f}, "
            f"{metrics['root_forward_backward_drift_m']:.3f}) m, root_height_range={metrics['root_height_range_m']:.3f} m, "
            f"lower_body_motion_ratio={metrics['lower_body_motion_ratio']:.3f}.",
            f"Gesture: left_hand_path={metrics['left_hand_path_m']:.3f} m, right_hand_path={metrics['right_hand_path_m']:.3f} m, "
            f"left_above_shoulder={metrics['left_hand_above_shoulder_ratio']:.3f}, "
            f"right_above_shoulder={metrics['right_hand_above_shoulder_ratio']:.3f}.",
            f"Problems: {problems}.",
            f"Suggested edit operations: {edits}.",
            f"Recommended canonical action: {metrics['recommended_action']}.",
            "Instruction to motion-editing LLM: this is generic reference-quality evidence unless action type is not unknown. "
            "Do not infer a semantic goal from the filename. Edit semantic content only when the user evaluation provides a clear goal.",
        ]
    )


def markdown_report(metrics: dict[str, Any], llm: str) -> str:
    rows = "\n".join(
        f"| {v['name']} | {v['value']:.4f} | {v['relation']} | {v['threshold']:.4f} | {v['severity']} |"
        for v in metrics["violations"]
    ) or "| none | - | - | - | - |"
    return f"""# Canonical Motion Quality Report: {metrics['motion_id']}

## Summary

| Metric | Value |
|---|---:|
| action type | {metrics['action_type'] or 'unknown'} |
| frames | {metrics['num_frames']} |
| duration sec | {metrics['duration_sec']:.3f} |
| max keypoint speed m/s | {metrics['max_keypoint_speed_mps']:.3f} |
| max keypoint accel m/s^2 | {metrics['max_keypoint_accel_mps2']:.3f} |
| max keypoint jerk m/s^3 | {metrics['max_keypoint_jerk_mps3']:.3f} |
| spike count | {metrics['spike_count']} |
| root path length m | {metrics['root_path_length_m']:.3f} |
| lower body motion ratio | {metrics['lower_body_motion_ratio']:.3f} |
| left hand path m | {metrics['left_hand_path_m']:.3f} |
| right hand path m | {metrics['right_hand_path_m']:.3f} |
| recommended action | {metrics['recommended_action']} |

## Violations

| Name | Value | Relation | Threshold | Severity |
|---|---:|---|---:|---|
{rows}

## Suggested Edit Operations

```text
{chr(10).join(metrics['suggested_edit_ops']) if metrics['suggested_edit_ops'] else 'none'}
```

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
        (output_dir / f"{motion_id}_canonical_quality.json").write_text(
            json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{motion_id}_canonical_llm.txt").write_text(llm + "\n", encoding="utf-8")
        (output_dir / f"{motion_id}_canonical_report.md").write_text(markdown_report(metrics, llm), encoding="utf-8")
        rows.append(
            {
                "motion_id": motion_id,
                "action_type": metrics["action_type"] or "",
                "num_frames": metrics["num_frames"],
                "duration_sec": metrics["duration_sec"],
                "max_keypoint_speed_mps": metrics["max_keypoint_speed_mps"],
                "max_keypoint_accel_mps2": metrics["max_keypoint_accel_mps2"],
                "max_keypoint_jerk_mps3": metrics["max_keypoint_jerk_mps3"],
                "spike_count": metrics["spike_count"],
                "root_path_length_m": metrics["root_path_length_m"],
                "lower_body_motion_ratio": metrics["lower_body_motion_ratio"],
                "violation_count": len(metrics["violations"]),
                "recommended_action": metrics["recommended_action"],
            }
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(to_jsonable(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "llm_batch_prompt.txt").write_text(
        "Canonical/reference quality summaries for a motion-editing LLM.\n\n"
        + "\n\n---\n\n".join(llm_blocks)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rules = load_rules(args.rules)
    if args.input_npz:
        paths = [args.input_npz]
    else:
        paths = sorted(p for p in args.input_dir.glob(args.glob) if p.is_file())
    if not paths:
        raise FileNotFoundError("No canonical npz files matched.")
    metrics_list = [
        compute_metrics(
            path,
            rules,
            action_label=args.action_label,
            infer_action_from_name=args.infer_action_from_name,
        )
        for path in paths
    ]
    write_outputs(metrics_list, args.output_dir)
    print(f"Wrote {len(metrics_list)} canonical quality reports to {args.output_dir}")


if __name__ == "__main__":
    main()
