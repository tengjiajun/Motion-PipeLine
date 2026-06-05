#!/usr/bin/env python
"""Apply an LLM edit plan to a canonical_motion.npz reference.

This script turns a structured edit plan into conservative numeric edits on the
canonical reference. It does not edit backend-specific FRoM-W1/ExBody files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SMPL24_LOWER = [1, 2, 4, 5, 7, 8, 10, 11]
SMPL24_ARMS = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
MOTIONX52_LOWER = [1, 2, 4, 5, 7, 8, 10, 11]
MOTIONX52_ARMS = list(range(13, 52))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply LLM edit plan to canonical motion.")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--edit-plan", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/canonical/canonical_llm_edited"),
    )
    parser.add_argument("--suffix", default="_canonical_motion_llm_v3.npz")
    parser.add_argument(
        "--max-time-stretch",
        type=float,
        default=1.35,
        help="Maximum duration multiplier allowed for slow_down/reduce_speed.",
    )
    parser.add_argument(
        "--base-smooth-window",
        type=int,
        default=5,
        help="Base temporal smoothing window; raised slightly for stronger smooth edits.",
    )
    return parser.parse_args()


def clean_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_canonical_motion_llm_v3", "_canonical_motion_v2", "_canonical_motion"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_npz_items(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {key: data[key] for key in data.files}


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.astype(np.float32).copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    flat = arr.reshape(arr.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(flat, dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out.reshape(arr.shape).astype(np.float32)


def blend(original: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    return (original * (1.0 - strength) + target * strength).astype(np.float32)


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return (quat / norm).astype(np.float32)


def quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    q = normalize_quat_xyzw(quat).reshape(-1, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out.reshape(quat.shape[:-1] + (3, 3)).astype(np.float32)


def resample_time(arr: np.ndarray, new_len: int) -> np.ndarray:
    old_len = arr.shape[0]
    if new_len == old_len:
        return arr.astype(np.float32).copy()
    old_t = np.linspace(0.0, 1.0, old_len, dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
    flat = arr.reshape(old_len, -1)
    out = np.empty((new_len, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(new_t, old_t, flat[:, col])
    return out.reshape((new_len,) + arr.shape[1:]).astype(np.float32)


def target_indices(target: str, kind: str) -> list[int] | None:
    target = (target or "").lower()
    if kind == "smpl24":
        if target in {"arms", "upper_body"}:
            return SMPL24_ARMS
        if target == "lower_body":
            return SMPL24_LOWER
    if kind == "motionx52":
        if target in {"arms", "upper_body"}:
            return MOTIONX52_ARMS
        if target == "lower_body":
            return MOTIONX52_LOWER
    return None


def smooth_array_by_target(arr: np.ndarray, indices: list[int] | None, window: int, strength: float) -> np.ndarray:
    smoothed = moving_average(arr, window)
    if indices is None:
        return blend(arr, smoothed, strength)
    out = arr.astype(np.float32).copy()
    out[:, indices] = blend(arr[:, indices], smoothed[:, indices], strength)
    return out


def stabilize_lower_body(arr: np.ndarray, lower_indices: list[int], strength: float) -> np.ndarray:
    if strength <= 0:
        return arr.astype(np.float32).copy()
    out = arr.astype(np.float32).copy()
    pelvis = arr[:, 0:1, :]
    rel0 = arr[0:1, lower_indices, :] - arr[0:1, 0:1, :]
    target = pelvis + rel0
    out[:, lower_indices] = blend(out[:, lower_indices], target, strength)
    return out


def reduce_joint_amplitude(arr: np.ndarray, indices: list[int] | None, strength: float) -> np.ndarray:
    if indices is None or strength <= 0:
        return arr.astype(np.float32).copy()
    out = arr.astype(np.float32).copy()
    # Conservative damping: only pull selected joints slightly toward their first
    # frame pose. This reduces overshoot without inventing a new semantic pose.
    damping = min(0.25, 0.5 * float(np.clip(strength, 0.0, 1.0)))
    target = np.repeat(arr[0:1, indices, :], arr.shape[0], axis=0)
    out[:, indices] = blend(out[:, indices], target, damping)
    return out


def summarize_edits(plan: dict[str, Any]) -> list[dict[str, Any]]:
    edits = plan.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("edit plan must contain a list field named 'edits'")
    return edits


def max_strength(edits: list[dict[str, Any]], edit_types: set[str]) -> float:
    values = [
        float(edit.get("strength", 0.0))
        for edit in edits
        if str(edit.get("type", "")).lower() in edit_types
    ]
    return max(values) if values else 0.0


def append_edit_log(items: dict[str, Any], entry: dict[str, Any]) -> None:
    try:
        old_log = json.loads(str(items.get("edit_log_json", np.asarray("[]"))))
    except json.JSONDecodeError:
        old_log = []
    old_log.append(entry)
    items["edit_log_json"] = np.asarray(json.dumps(old_log, ensure_ascii=False))


def apply_plan(input_npz: Path, output_npz: Path, plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(str(input_npz), allow_pickle=False)
    items = copy_npz_items(data)
    edits = summarize_edits(plan)

    original_len = int(items["num_frames"])
    speed_strength = max_strength(edits, {"slow_down", "reduce_speed"})
    time_stretch = min(float(args.max_time_stretch), 1.0 + 0.75 * speed_strength)
    new_len = max(original_len, int(round(original_len * time_stretch)))

    time_keys = [
        "smpl24_joints",
        "smpl24_struct_joints",
        "smpl29_joints",
        "root_translation",
        "motionx52_joints",
        "motionx52_confidence",
        "smpl24_local_quat_xyzw",
    ]
    for key in time_keys:
        items[key] = resample_time(items[key].astype(np.float32), new_len)
    items["smpl24_local_quat_xyzw"] = normalize_quat_xyzw(items["smpl24_local_quat_xyzw"])

    applied_ops: list[dict[str, Any]] = []
    for edit in edits:
        edit_type = str(edit.get("type", "")).lower()
        target = str(edit.get("target", "whole_body")).lower()
        strength = float(np.clip(float(edit.get("strength", 0.0)), 0.0, 1.0))
        if strength <= 0:
            continue

        if edit_type == "smooth":
            window = max(args.base_smooth_window, 3 + int(round(strength * 6)))
            if window % 2 == 0:
                window += 1
            for key, kind in (
                ("smpl24_joints", "smpl24"),
                ("smpl24_struct_joints", "smpl24"),
                ("smpl29_joints", "smpl24"),
                ("motionx52_joints", "motionx52"),
            ):
                items[key] = smooth_array_by_target(
                    items[key].astype(np.float32),
                    target_indices(target, kind),
                    window,
                    strength,
                )
            items["root_translation"] = smooth_array_by_target(
                items["root_translation"].astype(np.float32),
                None if target in {"whole_body", "root_translation"} else [],
                window,
                min(strength, 0.35),
            )
            quat = items["smpl24_local_quat_xyzw"].astype(np.float32)
            quat_smoothed = normalize_quat_xyzw(moving_average(quat, window))
            items["smpl24_local_quat_xyzw"] = normalize_quat_xyzw(blend(quat, quat_smoothed, min(strength, 0.5)))
            applied_ops.append({"type": edit_type, "target": target, "strength": strength, "window": window})

        elif edit_type == "stabilize_lower_body":
            items["smpl24_joints"] = stabilize_lower_body(items["smpl24_joints"], SMPL24_LOWER, strength)
            items["smpl24_struct_joints"] = stabilize_lower_body(items["smpl24_struct_joints"], SMPL24_LOWER, strength)
            items["smpl29_joints"] = stabilize_lower_body(items["smpl29_joints"], SMPL24_LOWER, strength)
            items["motionx52_joints"] = stabilize_lower_body(items["motionx52_joints"], MOTIONX52_LOWER, strength)
            applied_ops.append({"type": edit_type, "target": "lower_body", "strength": strength})

        elif edit_type == "stabilize_root":
            root = items["root_translation"].astype(np.float32)
            target_root = root.copy()
            target_root[:, 0] = root[0, 0]
            target_root[:, 1] = root[0, 1]
            items["root_translation"] = blend(root, target_root, min(strength, 0.5))
            applied_ops.append({"type": edit_type, "target": "root_translation", "strength": strength})

        elif edit_type == "reduce_amplitude":
            for key, kind in (
                ("smpl24_joints", "smpl24"),
                ("smpl24_struct_joints", "smpl24"),
                ("smpl29_joints", "smpl24"),
                ("motionx52_joints", "motionx52"),
            ):
                items[key] = reduce_joint_amplitude(
                    items[key].astype(np.float32),
                    target_indices(target, kind),
                    strength,
                )
            applied_ops.append(
                {
                    "type": edit_type,
                    "target": target,
                    "strength": strength,
                    "note": "conservative damping toward first-frame pose",
                }
            )

        elif edit_type in {"slow_down", "reduce_speed"}:
            # Already applied globally through time_stretch.
            applied_ops.append({"type": edit_type, "target": target, "strength": strength, "time_stretch": time_stretch})

    items["smpl24_local_quat_xyzw"] = normalize_quat_xyzw(items["smpl24_local_quat_xyzw"].astype(np.float32))
    items["smpl24_local_rotmat"] = quat_xyzw_to_rotmat(items["smpl24_local_quat_xyzw"])
    items["num_frames"] = np.asarray(new_len, dtype=np.int32)

    metadata = json.loads(str(items["metadata_json"]))
    metadata["canonical_format"] = "canonical_motion_v1_llm_edited"
    metadata["llm_edited_from"] = str(input_npz)
    metadata["llm_edit_plan_path"] = str(args.edit_plan)
    metadata["llm_time_stretch"] = time_stretch
    metadata["llm_num_frames_before"] = original_len
    metadata["llm_num_frames_after"] = new_len
    items["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))

    log_entry = {
        "operation": "apply_llm_edit_plan",
        "input_npz": str(input_npz),
        "edit_plan": str(args.edit_plan),
        "original_num_frames": original_len,
        "new_num_frames": new_len,
        "time_stretch": time_stretch,
        "applied_ops": applied_ops,
    }
    append_edit_log(items, log_entry)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_npz), **items)
    return log_entry


def main() -> None:
    args = parse_args()
    input_npz = args.input_npz.resolve()
    plan = load_json(args.edit_plan.resolve())
    output_npz = (
        args.output_npz.resolve()
        if args.output_npz is not None
        else (args.output_dir / f"{clean_stem(input_npz)}{args.suffix}").resolve()
    )

    log_entry = apply_plan(input_npz, output_npz, plan, args)
    log_path = output_npz.with_suffix(".edit_applied.json")
    log_path.write_text(json.dumps(log_entry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {input_npz} -> {output_npz}")
    print(f"[LOG] {log_path}")


if __name__ == "__main__":
    main()
