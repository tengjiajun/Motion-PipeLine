#!/usr/bin/env python
"""Generic repair for canonical_motion.npz references.

This is a reference-level edit, not AlphaPose repair and not backend conversion.
It applies conservative temporal smoothing and lower-body stabilization while
keeping semantic limb motion as much as possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SMPL24_LOWER = [1, 2, 4, 5, 7, 8, 10, 11]
MOTIONX52_LOWER = [1, 2, 4, 5, 7, 8, 10, 11]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair canonical_motion.npz with generic smoothing/stabilization.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*_canonical_motion.npz")
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/canonical_edited"))
    parser.add_argument("--suffix", default="_canonical_motion_v2.npz")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--upper-smooth-strength", type=float, default=0.35)
    parser.add_argument("--lower-smooth-strength", type=float, default=0.65)
    parser.add_argument("--lower-stabilize-strength", type=float, default=0.45)
    parser.add_argument("--root-stabilize-strength", type=float, default=0.25)
    return parser.parse_args()


def clean_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_canonical_motion_v2", "_canonical_motion"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_npz:
        return [args.input_npz.resolve()]
    return sorted(p.resolve() for p in args.input_dir.glob(args.glob) if p.is_file())


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    flat = arr.reshape(arr.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(flat)
    for col in range(flat.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out.reshape(arr.shape)


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
    return out.reshape(quat.shape[:-1] + (3, 3))


def smooth_joint_array(arr: np.ndarray, lower_indices: list[int], window: int, upper_strength: float, lower_strength: float) -> np.ndarray:
    smoothed = moving_average(arr, window)
    out = blend(arr, smoothed, upper_strength)
    if lower_indices:
        out[:, lower_indices] = blend(arr[:, lower_indices], smoothed[:, lower_indices], lower_strength)
    return out.astype(np.float32)


def stabilize_lower_body(arr: np.ndarray, lower_indices: list[int], strength: float) -> np.ndarray:
    if not lower_indices or strength <= 0:
        return arr.astype(np.float32)
    out = arr.copy()
    pelvis = arr[:, 0:1, :]
    rel0 = arr[0:1, lower_indices, :] - arr[0:1, 0:1, :]
    target = pelvis + rel0
    out[:, lower_indices] = blend(out[:, lower_indices], target, strength)
    return out.astype(np.float32)


def stabilize_root_translation(root: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return root.astype(np.float32)
    target = root.copy()
    target[:, 0] = root[0, 0]
    target[:, 1] = root[0, 1]
    return blend(root, target, strength)


def copy_npz_items(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {key: data[key] for key in data.files}


def repair_one(input_npz: Path, output_npz: Path, args: argparse.Namespace) -> None:
    data = np.load(str(input_npz), allow_pickle=False)
    items = copy_npz_items(data)

    smpl24 = items["smpl24_joints"].astype(np.float32)
    smpl24_struct = items["smpl24_struct_joints"].astype(np.float32)
    smpl29 = items["smpl29_joints"].astype(np.float32)
    motionx52 = items["motionx52_joints"].astype(np.float32)

    smpl24_edit = smooth_joint_array(
        smpl24,
        SMPL24_LOWER,
        args.smooth_window,
        args.upper_smooth_strength,
        args.lower_smooth_strength,
    )
    smpl24_struct_edit = smooth_joint_array(
        smpl24_struct,
        SMPL24_LOWER,
        args.smooth_window,
        args.upper_smooth_strength,
        args.lower_smooth_strength,
    )
    smpl29_edit = smooth_joint_array(
        smpl29,
        SMPL24_LOWER,
        args.smooth_window,
        args.upper_smooth_strength,
        args.lower_smooth_strength,
    )
    motionx52_edit = smooth_joint_array(
        motionx52,
        MOTIONX52_LOWER,
        args.smooth_window,
        args.upper_smooth_strength,
        args.lower_smooth_strength,
    )

    smpl24_edit = stabilize_lower_body(smpl24_edit, SMPL24_LOWER, args.lower_stabilize_strength)
    smpl24_struct_edit = stabilize_lower_body(smpl24_struct_edit, SMPL24_LOWER, args.lower_stabilize_strength)
    smpl29_edit = stabilize_lower_body(smpl29_edit, SMPL24_LOWER, args.lower_stabilize_strength)
    motionx52_edit = stabilize_lower_body(motionx52_edit, MOTIONX52_LOWER, args.lower_stabilize_strength)

    root_translation = stabilize_root_translation(items["root_translation"].astype(np.float32), args.root_stabilize_strength)

    quat = items["smpl24_local_quat_xyzw"].astype(np.float32)
    quat_smooth = normalize_quat_xyzw(moving_average(quat, args.smooth_window))
    quat_edit = normalize_quat_xyzw(blend(quat, quat_smooth, args.upper_smooth_strength))
    quat_edit[:, SMPL24_LOWER] = normalize_quat_xyzw(
        blend(quat[:, SMPL24_LOWER], quat_smooth[:, SMPL24_LOWER], args.lower_smooth_strength)
    )
    rotmat_edit = quat_xyzw_to_rotmat(quat_edit)

    edit_entry = {
        "operation": "generic_canonical_repair",
        "smooth_window": int(args.smooth_window),
        "upper_smooth_strength": float(args.upper_smooth_strength),
        "lower_smooth_strength": float(args.lower_smooth_strength),
        "lower_stabilize_strength": float(args.lower_stabilize_strength),
        "root_stabilize_strength": float(args.root_stabilize_strength),
        "targets": [
            "smpl24_joints",
            "smpl24_struct_joints",
            "smpl29_joints",
            "motionx52_joints",
            "smpl24_local_quat_xyzw",
            "root_translation",
        ],
    }
    old_log = json.loads(str(items.get("edit_log_json", np.asarray("[]"))))
    old_log.append(edit_entry)

    metadata = json.loads(str(items["metadata_json"]))
    metadata["canonical_format"] = "canonical_motion_v1_edited"
    metadata["edited_from"] = str(input_npz)
    metadata["generic_repair"] = edit_entry

    items["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    items["source_smpl_path"] = items["source_smpl_path"]
    items["smpl24_joints"] = smpl24_edit
    items["smpl24_struct_joints"] = smpl24_struct_edit
    items["smpl29_joints"] = smpl29_edit
    items["motionx52_joints"] = motionx52_edit
    items["root_translation"] = root_translation
    items["smpl24_local_quat_xyzw"] = quat_edit
    items["smpl24_local_rotmat"] = rotmat_edit
    items["edit_log_json"] = np.asarray(json.dumps(old_log, ensure_ascii=False))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_npz), **items)
    print(f"[DONE] {input_npz} -> {output_npz}")


def main() -> None:
    args = parse_args()
    inputs = iter_inputs(args)
    if not inputs:
        raise FileNotFoundError("No canonical npz files matched.")
    if args.output_npz and len(inputs) != 1:
        raise ValueError("--output-npz can only be used with one input.")
    for input_npz in inputs:
        output_npz = args.output_npz or args.output_dir / f"{clean_stem(input_npz)}{args.suffix}"
        repair_one(input_npz, output_npz.resolve(), args)


if __name__ == "__main__":
    main()
