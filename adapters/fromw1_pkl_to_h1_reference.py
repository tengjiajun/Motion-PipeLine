#!/usr/bin/env python
"""Convert FRoM-W1 retargeted H1 pkl files to h1_reference_motion.npz.

The output is a robot-level reference motion. It is intentionally downstream of
human/SMPL canonical data and stores H1 root and 19-DoF joint references as the
primary editable representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


H1_DOF_NAMES = np.asarray(
    [
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_joint",
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FRoM-W1 H1 pkl to h1_reference_motion.npz.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-pkl", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*.pkl")
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/h1_reference/fromw1"))
    parser.add_argument("--motion-id", default=None)
    parser.add_argument("--backend-tag", default="fromw1")
    parser.add_argument("--suffix", default="_h1_reference_motion.npz")
    return parser.parse_args()


def clean_motion_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_h1_reference_motion", "_623"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_pkl:
        return [args.input_pkl.resolve()]
    return sorted(p.resolve() for p in args.input_dir.glob(args.glob) if p.is_file())


def load_fromw1_motion(path: Path) -> tuple[str, dict[str, Any]]:
    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid FRoM-W1 pkl: {path}")
    if "motion 0" in data:
        return "motion 0", data["motion 0"]
    key = next(iter(data.keys()))
    motion = data[key]
    if not isinstance(motion, dict):
        raise ValueError(f"Unsupported FRoM-W1 pkl structure: {path}")
    return str(key), motion


def finite_difference(arr: np.ndarray, fps: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.shape[0] <= 1:
        return np.zeros_like(arr, dtype=np.float32)
    dt = 1.0 / max(float(fps), 1e-6)
    return np.gradient(arr, dt, axis=0).astype(np.float32)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1).astype(np.float32)


def validate_motion(path: Path, motion: dict[str, Any]) -> None:
    required = ["root_trans_offset", "root_rot", "dof", "fps"]
    missing = [key for key in required if key not in motion]
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    dof = np.asarray(motion["dof"])
    if dof.ndim != 2 or dof.shape[1] != 19:
        raise ValueError(f"{path} dof shape must be (T, 19), got {dof.shape}")
    root = np.asarray(motion["root_trans_offset"])
    root_rot = np.asarray(motion["root_rot"])
    if root.shape != (dof.shape[0], 3):
        raise ValueError(f"{path} root_trans_offset shape {root.shape}, expected {(dof.shape[0], 3)}")
    if root_rot.shape != (dof.shape[0], 4):
        raise ValueError(f"{path} root_rot shape {root_rot.shape}, expected {(dof.shape[0], 4)}")


def convert_one(input_pkl: Path, output_npz: Path, motion_id: str | None, backend_tag: str) -> None:
    motion_key, motion = load_fromw1_motion(input_pkl)
    validate_motion(input_pkl, motion)

    fps = int(motion.get("fps", 30))
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion["dof"], dtype=np.float32)
    pose_aa = np.asarray(motion.get("pose_aa", np.zeros((dof_pos.shape[0], 0, 3))), dtype=np.float32)
    hand_pose = np.asarray(motion.get("hand_pose", np.zeros((dof_pos.shape[0], 0))), dtype=np.float32)
    body_names = np.asarray(motion.get("body_names", []))

    resolved_motion_id = motion_id or clean_motion_id(input_pkl)
    metadata = {
        "format": "h1_reference_motion_v1",
        "motion_id": resolved_motion_id,
        "source_backend": backend_tag,
        "source_format": "fromw1_h1_pkl",
        "source_pkl_path": str(input_pkl),
        "source_motion_key": motion_key,
        "primary_representation": "h1_dof_pos_ref",
        "root_quat_order": "xyzw",
        "dof_units": "radian",
        "position_units": "meter",
        "note": "Robot-level reference motion before environment execution.",
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_npz),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        motion_id=np.asarray(resolved_motion_id),
        source_backend=np.asarray(backend_tag),
        source_pkl_path=np.asarray(str(input_pkl)),
        fps=np.asarray(fps, dtype=np.int32),
        num_frames=np.asarray(dof_pos.shape[0], dtype=np.int32),
        h1_joint_names=H1_DOF_NAMES,
        h1_body_names=body_names,
        root_pos_ref=root_pos,
        root_vel_ref=finite_difference(root_pos, fps),
        root_quat_ref_xyzw=root_quat_xyzw,
        root_quat_ref_wxyz=quat_xyzw_to_wxyz(root_quat_xyzw),
        h1_dof_pos_ref=dof_pos,
        h1_dof_vel_ref=finite_difference(dof_pos, fps),
        h1_body_pose_aa_ref=pose_aa,
        hand_pose_ref=hand_pose,
    )
    print(f"[DONE] {input_pkl} -> {output_npz} frames={dof_pos.shape[0]} fps={fps}")


def main() -> None:
    args = parse_args()
    inputs = iter_inputs(args)
    if not inputs:
        raise FileNotFoundError("No FRoM-W1 pkl files matched.")
    if args.output_npz and len(inputs) != 1:
        raise ValueError("--output-npz can only be used with one input.")

    for input_pkl in inputs:
        motion_id = args.motion_id if len(inputs) == 1 else clean_motion_id(input_pkl)
        output_npz = args.output_npz or args.output_dir / f"{motion_id}{args.suffix}"
        convert_one(
            input_pkl=input_pkl,
            output_npz=output_npz.resolve(),
            motion_id=motion_id,
            backend_tag=args.backend_tag,
        )


if __name__ == "__main__":
    main()
