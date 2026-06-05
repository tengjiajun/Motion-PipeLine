#!/usr/bin/env python
"""Convert repaired AlphaPose SMPL npy files into canonical_motion.npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SMPL24_NAMES = np.asarray(
    [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ]
)

SMPL29_NAMES = np.asarray(
    [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
    ]
)

MOTIONX52_NAMES = np.asarray(
    [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert repaired AlphaPose SMPL npy to canonical npz.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npy", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*_smpl_repaired_compact.npy")
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/canonical"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use-transl", action="store_true", help="Add transl to joint positions in canonical joints.")
    parser.add_argument("--no-cv-y-flip", action="store_true", help="Do not build camera-converted MotionX Y axis.")
    parser.add_argument("--no-cv-z-flip", action="store_true", help="Do not build camera-converted MotionX Z axis.")
    return parser.parse_args()


def motion_id_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_smpl_repaired_compact", "_smpl_repaired", "_smpl_raw"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_payload(path: Path) -> dict[str, Any]:
    obj = np.load(str(path), allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict) or "frames" not in obj:
        raise ValueError(f"Unsupported SMPL npy payload: {path}")
    return obj


def pick_person(frame: dict[str, Any]) -> dict[str, Any]:
    people = frame.get("result") or []
    if not people:
        raise ValueError("Canonical input should already be repaired/compact and contain one person per frame.")
    return max(people, key=lambda item: float(item.get("bbox_score", 0.0)))


def rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float32)
    flat = rot.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=np.float32)
    for i, m in enumerate(flat):
        trace = float(np.trace(m))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        else:
            idx = int(np.argmax(np.diag(m)))
            if idx == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                qw = (m[2, 1] - m[1, 2]) / s
                qx = 0.25 * s
                qy = (m[0, 1] + m[1, 0]) / s
                qz = (m[0, 2] + m[2, 0]) / s
            elif idx == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                qw = (m[0, 2] - m[2, 0]) / s
                qx = (m[0, 1] + m[1, 0]) / s
                qy = 0.25 * s
                qz = (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                qw = (m[1, 0] - m[0, 1]) / s
                qx = (m[0, 2] + m[2, 0]) / s
                qy = (m[1, 2] + m[2, 1]) / s
                qz = 0.25 * s
        quat = np.asarray([qx, qy, qz, qw], dtype=np.float32)
        quat /= max(float(np.linalg.norm(quat)), 1e-8)
        out[i] = quat
    return out.reshape(rot.shape[:-2] + (4,))


def build_hand_template(is_right: bool) -> np.ndarray:
    sign = -1.0 if is_right else 1.0
    finger_dirs = {
        "index": np.array([sign, 0.12, 0.05], dtype=np.float32),
        "middle": np.array([sign, 0.00, 0.07], dtype=np.float32),
        "pinky": np.array([sign, -0.18, 0.02], dtype=np.float32),
        "ring": np.array([sign, -0.10, 0.05], dtype=np.float32),
        "thumb": np.array([sign * 0.70, 0.22, -0.10], dtype=np.float32),
    }
    segment_lengths = [0.026, 0.020, 0.016]
    out = []
    for name in ["index", "middle", "pinky", "ring", "thumb"]:
        direction = finger_dirs[name] / (np.linalg.norm(finger_dirs[name]) + 1e-8)
        point = direction * 0.035
        for length in segment_lengths:
            point = point + direction * length
            out.append(point.copy())
    return np.asarray(out, dtype=np.float32)


SMPL_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def local_to_global_rot(local24: np.ndarray) -> np.ndarray:
    out = np.zeros_like(local24, dtype=np.float32)
    for i, parent in enumerate(SMPL_PARENT):
        out[i] = local24[i] if parent < 0 else out[parent] @ local24[i]
    return out


def make_motionx52(body24: np.ndarray, theta: np.ndarray) -> np.ndarray:
    t = body24.shape[0]
    joints = np.zeros((t, 52, 3), dtype=np.float32)
    joints[:, :22] = body24[:, :22]
    hand_l = build_hand_template(is_right=False)
    hand_r = build_hand_template(is_right=True)
    for i in range(t):
        grot = local_to_global_rot(theta[i])
        joints[i, 22:37] = joints[i, 20] + (grot[20] @ hand_l.T).T
        joints[i, 37:52] = joints[i, 21] + (grot[21] @ hand_r.T).T
    return joints


def convert_one(input_npy: Path, output_npz: Path, fps: int, use_transl: bool, cv_y_flip: bool, cv_z_flip: bool) -> None:
    payload = load_payload(input_npy)
    people = [pick_person(frame) for frame in payload["frames"]]
    motion_id = motion_id_from_path(input_npy)

    smpl24_joints = np.stack([np.asarray(p["pred_xyz_jts_24"], dtype=np.float32) for p in people])
    smpl24_struct_joints = np.stack([np.asarray(p["pred_xyz_jts_24_struct"], dtype=np.float32) for p in people])
    smpl29_joints = np.stack([np.asarray(p["pred_xyz_jts_29"], dtype=np.float32) for p in people])
    smpl24_local_rotmat = np.stack([np.asarray(p["pred_theta_mats"], dtype=np.float32) for p in people])
    smpl24_local_quat_xyzw = rotmat_to_quat_xyzw(smpl24_local_rotmat)
    root_translation = np.stack([np.asarray(p["transl"], dtype=np.float32).reshape(3) for p in people])
    body_shape = np.median(np.stack([np.asarray(p["pred_shape"], dtype=np.float32) for p in people]), axis=0).astype(np.float32)
    kp_score = np.stack([np.asarray(p["kp_score"], dtype=np.float32).reshape(-1) for p in people])

    body24_for_motionx = smpl24_joints.copy()
    if use_transl:
        body24_for_motionx = body24_for_motionx + root_translation[:, None, :]
    motionx52_joints = make_motionx52(body24_for_motionx, smpl24_local_rotmat)
    if cv_y_flip:
        motionx52_joints[..., 1] *= -1.0
    if cv_z_flip:
        motionx52_joints[..., 2] *= -1.0

    motionx52_confidence = np.ones((len(people), 52), dtype=np.float32)
    motionx52_confidence[:, :29] = np.pad(kp_score[:, :29], ((0, 0), (0, 0)), constant_values=1.0)

    metadata = {
        "motion_id": motion_id,
        "fps": int(fps),
        "num_frames": int(len(people)),
        "source_smpl_path": str(input_npy),
        "source_format": payload.get("format", ""),
        "canonical_format": "canonical_motion_v1",
        "root_translation_source": "transl",
        "motionx52": {
            "source": "smpl24_joints_plus_synthetic_hands",
            "use_transl": bool(use_transl),
            "cv_y_flip": bool(cv_y_flip),
            "cv_z_flip": bool(cv_z_flip),
        },
    }
    if "repair_meta" in payload:
        metadata["repair_meta"] = payload["repair_meta"]
    if "compact_meta" in payload:
        metadata["compact_meta"] = payload["compact_meta"]

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_npz),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        motion_id=np.asarray(motion_id),
        fps=np.asarray(fps, dtype=np.int32),
        num_frames=np.asarray(len(people), dtype=np.int32),
        source_smpl_path=np.asarray(str(input_npy)),
        smpl24_joint_names=SMPL24_NAMES,
        smpl29_joint_names=SMPL29_NAMES,
        motionx52_joint_names=MOTIONX52_NAMES,
        smpl24_local_rotmat=smpl24_local_rotmat,
        smpl24_local_quat_xyzw=smpl24_local_quat_xyzw,
        smpl24_joints=smpl24_joints,
        smpl24_struct_joints=smpl24_struct_joints,
        smpl29_joints=smpl29_joints,
        root_translation=root_translation,
        body_shape=body_shape,
        motionx52_joints=motionx52_joints,
        motionx52_confidence=motionx52_confidence,
        edit_log_json=np.asarray("[]"),
        quality_json=np.asarray("{}"),
    )
    print(f"[DONE] {input_npy} -> {output_npz}")


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_npy:
        return [args.input_npy.resolve()]
    return sorted(p.resolve() for p in args.input_dir.glob(args.glob) if p.is_file() and p.stat().st_size > 0)


def main() -> None:
    args = parse_args()
    inputs = iter_inputs(args)
    if not inputs:
        raise FileNotFoundError("No input npy files matched.")
    if args.output_npz and len(inputs) != 1:
        raise ValueError("--output-npz can only be used with one input.")

    for input_npy in inputs:
        output_npz = args.output_npz or args.output_dir / f"{motion_id_from_path(input_npy)}_canonical_motion.npz"
        convert_one(
            input_npy=input_npy,
            output_npz=output_npz.resolve(),
            fps=args.fps,
            use_transl=args.use_transl,
            cv_y_flip=not args.no_cv_y_flip,
            cv_z_flip=not args.no_cv_z_flip,
        )


if __name__ == "__main__":
    main()
