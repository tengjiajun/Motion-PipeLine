#!/usr/bin/env python
"""Convert canonical_motion.npz to ExBody/Isaac Gym H1 retarget files.

Run this adapter in the ExBody environment where poselib and the retarget config
assets are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ALPHAPOSE_TO_POSELIB_SMPL = [
    0,
    1,
    4,
    7,
    10,
    2,
    5,
    8,
    11,
    3,
    6,
    9,
    12,
    15,
    13,
    16,
    18,
    20,
    22,
    14,
    17,
    19,
    21,
    23,
]

ROTATION_PRESETS = {
    "identity": (0.0, 0.0, 0.0),
    "x90": (90.0, 0.0, 0.0),
    "x-90": (-90.0, 0.0, 0.0),
    "x180": (180.0, 0.0, 0.0),
    "y90": (0.0, 90.0, 0.0),
    "y-90": (0.0, -90.0, 0.0),
    "y180": (0.0, 180.0, 0.0),
    "z90": (0.0, 0.0, 90.0),
    "z-90": (0.0, 0.0, -90.0),
    "z180": (0.0, 0.0, 180.0),
}

H1_KEY_BODY_NAMES = [
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_link",
    "left_shoulder_pitch_link",
    "left_elbow_link",
    "left_hand_link",
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_hand_link",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert canonical_motion.npz to ExBody H1 motion files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*_canonical_motion.npz")
    parser.add_argument(
        "--exbody-root",
        type=Path,
        required=True,
        help="ExBody/expressive-humanoid root containing data/configs and data/tpose.",
    )
    parser.add_argument("--source-output-dir", type=Path, default=Path("data/exbody_inputs/npy"))
    parser.add_argument("--retarget-output-dir", type=Path, default=Path("data/exbody_inputs/retarget_npy"))
    parser.add_argument("--fps", type=int, default=None, help="Override canonical fps.")
    parser.add_argument("--use-root-translation", action="store_true")
    parser.add_argument("--no-center-xy", action="store_true")
    parser.add_argument("--rotation-preset", default="identity", choices=sorted(ROTATION_PRESETS))
    parser.add_argument("--rotation-deg", nargs=3, type=float, metavar=("RX", "RY", "RZ"))
    parser.add_argument("--retarget-rotation-preset", default=None, choices=sorted(ROTATION_PRESETS))
    parser.add_argument("--retarget-rotation-deg", nargs=3, type=float, metavar=("RX", "RY", "RZ"))
    return parser.parse_args()


def import_poselib() -> dict[str, Any]:
    try:
        from poselib.core.rotation3d import (
            quat_from_angle_axis,
            quat_identity,
            quat_inverse,
            quat_mul,
            quat_rotate,
        )
        from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
    except ImportError as exc:
        raise SystemExit(
            "poselib is not importable. Run this adapter inside the ExBody/Isaac Gym environment."
        ) from exc
    return {
        "quat_from_angle_axis": quat_from_angle_axis,
        "quat_identity": quat_identity,
        "quat_inverse": quat_inverse,
        "quat_mul": quat_mul,
        "quat_rotate": quat_rotate,
        "SkeletonMotion": SkeletonMotion,
        "SkeletonState": SkeletonState,
    }


def clean_stem(path: Path) -> str:
    stem = path.stem
    suffix = "_canonical_motion"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_npz:
        return [args.input_npz.resolve()]
    return sorted(p.resolve() for p in args.input_dir.glob(args.glob) if p.is_file())


def quat_from_euler_deg(rx: float, ry: float, rz: float, plib: dict[str, Any]) -> torch.Tensor:
    quat_from_angle_axis = plib["quat_from_angle_axis"]
    quat_mul = plib["quat_mul"]
    qx = quat_from_angle_axis(
        torch.tensor([np.deg2rad(rx)], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )[0]
    qy = quat_from_angle_axis(
        torch.tensor([np.deg2rad(ry)], dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )[0]
    qz = quat_from_angle_axis(
        torch.tensor([np.deg2rad(rz)], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )[0]
    return quat_mul(qz.unsqueeze(0), quat_mul(qy.unsqueeze(0), qx.unsqueeze(0)))[0]


def build_source_motion(
    canonical: np.lib.npyio.NpzFile,
    exbody_root: Path,
    fps: int,
    use_root_translation: bool,
    center_xy: bool,
    correction_q: torch.Tensor,
    plib: dict[str, Any],
):
    SkeletonState = plib["SkeletonState"]
    SkeletonMotion = plib["SkeletonMotion"]
    quat_mul = plib["quat_mul"]
    quat_rotate = plib["quat_rotate"]
    quat_identity = plib["quat_identity"]

    source_tpose = SkeletonState.from_file(str(exbody_root / "data" / "tpose" / "smpl_tpose.npy"))
    local_rotation = torch.from_numpy(
        canonical["smpl24_local_quat_xyzw"][:, ALPHAPOSE_TO_POSELIB_SMPL, :].astype(np.float32)
    )
    if use_root_translation:
        root_translation = torch.from_numpy(canonical["root_translation"].astype(np.float32)).clone()
    else:
        root_translation = torch.zeros((local_rotation.shape[0], 3), dtype=torch.float32)

    if not torch.allclose(correction_q, quat_identity([1])[0], atol=1e-6):
        correction = correction_q.unsqueeze(0).expand(local_rotation.shape[0], -1)
        local_rotation[:, 0, :] = quat_mul(correction, local_rotation[:, 0, :])
        root_translation = quat_rotate(correction.unsqueeze(1), root_translation.unsqueeze(1)).squeeze(1)

    if center_xy:
        root_translation[:, 0] -= root_translation[0, 0]
        root_translation[:, 1] -= root_translation[0, 1]

    state = SkeletonState.from_rotation_and_root_translation(
        source_tpose.skeleton_tree,
        local_rotation,
        root_translation,
        is_local=True,
    )
    return SkeletonMotion.from_skeleton_state(state, fps=fps)


def retarget_to_h1(source_motion, exbody_root: Path, rotation_to_target_override: torch.Tensor | None, plib: dict[str, Any]):
    SkeletonState = plib["SkeletonState"]
    SkeletonMotion = plib["SkeletonMotion"]

    cfg_path = exbody_root / "data" / "configs" / "retarget_smpl_to_h1.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    source_tpose = SkeletonState.from_file(str(exbody_root / cfg["source_tpose"]))
    target_tpose = SkeletonState.from_file(str(exbody_root / cfg["target_tpose"]))
    rotation_to_target = (
        rotation_to_target_override.clone().float()
        if rotation_to_target_override is not None
        else torch.tensor(cfg["rotation"], dtype=torch.float32)
    )

    target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=cfg["joint_mapping"],
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target,
        scale_to_target_skeleton=cfg["scale"],
    )

    frame_beg = cfg.get("trim_frame_beg", -1)
    frame_end = cfg.get("trim_frame_end", -1)
    frame_beg = 0 if frame_beg == -1 else frame_beg
    frame_end = target_motion.local_rotation.shape[0] if frame_end == -1 else frame_end

    local_rotation = target_motion.local_rotation[frame_beg:frame_end].clone()
    root_translation = target_motion.root_translation[frame_beg:frame_end].clone()
    root_translation[:, 0] -= root_translation[0, 0]
    root_translation[:, 1] -= root_translation[0, 1]

    min_h = torch.min(target_motion.global_translation[frame_beg:frame_end, :, 2])
    root_translation[:, 2] += -min_h
    state = SkeletonState.from_rotation_and_root_translation(
        target_motion.skeleton_tree,
        local_rotation,
        root_translation,
        is_local=True,
    )
    return SkeletonMotion.from_skeleton_state(state, fps=target_motion.fps)


def build_key_bodies(target_motion, plib: dict[str, Any]) -> np.ndarray:
    quat_inverse = plib["quat_inverse"]
    quat_rotate = plib["quat_rotate"]

    node_indices = target_motion.skeleton_tree._node_indices
    key_body_ids = [node_indices[name] for name in H1_KEY_BODY_NAMES]
    global_pos = target_motion.global_translation[:, key_body_ids, :]
    root_pos = target_motion.root_translation[:, None, :]
    root_rot = target_motion.global_rotation[:, 0, :]
    root_rot_inv = quat_inverse(root_rot)[:, None, :]
    local_pos = quat_rotate(root_rot_inv.expand(-1, len(key_body_ids), -1), global_pos - root_pos)
    return local_pos.cpu().numpy().astype(np.float32)


def convert_one(
    input_npz: Path,
    exbody_root: Path,
    source_output_dir: Path,
    retarget_output_dir: Path,
    fps_override: int | None,
    use_root_translation: bool,
    center_xy: bool,
    correction_q: torch.Tensor,
    retarget_rotation_override: torch.Tensor | None,
    plib: dict[str, Any],
) -> None:
    canonical = np.load(str(input_npz), allow_pickle=False)
    fps = int(fps_override if fps_override is not None else int(canonical["fps"]))
    motion_name = clean_stem(input_npz)

    source_motion = build_source_motion(
        canonical=canonical,
        exbody_root=exbody_root,
        fps=fps,
        use_root_translation=use_root_translation,
        center_xy=center_xy,
        correction_q=correction_q,
        plib=plib,
    )
    target_motion = retarget_to_h1(
        source_motion=source_motion,
        exbody_root=exbody_root,
        rotation_to_target_override=retarget_rotation_override,
        plib=plib,
    )
    key_bodies = build_key_bodies(target_motion, plib=plib)

    source_output_dir.mkdir(parents=True, exist_ok=True)
    retarget_output_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_output_dir / f"{motion_name}.npy"
    target_path = retarget_output_dir / f"{motion_name}.npy"
    key_body_path = retarget_output_dir / f"{motion_name}_key_bodies.npy"
    source_motion.to_file(str(source_path))
    target_motion.to_file(str(target_path))
    np.save(str(key_body_path), key_bodies)
    print(f"[DONE] {input_npz} -> {source_path}")
    print(f"[DONE] {input_npz} -> {target_path}")
    print(f"[DONE] {input_npz} -> {key_body_path} shape={key_bodies.shape}")


def main() -> None:
    args = parse_args()
    plib = import_poselib()
    exbody_root = args.exbody_root.resolve()

    if args.rotation_deg is not None:
        correction_q = quat_from_euler_deg(*args.rotation_deg, plib=plib)
    else:
        correction_q = quat_from_euler_deg(*ROTATION_PRESETS[args.rotation_preset], plib=plib)

    if args.retarget_rotation_deg is not None:
        retarget_rotation_override = quat_from_euler_deg(*args.retarget_rotation_deg, plib=plib)
    elif args.retarget_rotation_preset is not None:
        retarget_rotation_override = quat_from_euler_deg(*ROTATION_PRESETS[args.retarget_rotation_preset], plib=plib)
    else:
        retarget_rotation_override = None

    inputs = iter_inputs(args)
    if not inputs:
        raise FileNotFoundError("No canonical npz files matched.")

    for input_npz in inputs:
        convert_one(
            input_npz=input_npz,
            exbody_root=exbody_root,
            source_output_dir=args.source_output_dir,
            retarget_output_dir=args.retarget_output_dir,
            fps_override=args.fps,
            use_root_translation=args.use_root_translation,
            center_xy=not args.no_center_xy,
            correction_q=correction_q,
            retarget_rotation_override=retarget_rotation_override,
            plib=plib,
        )


if __name__ == "__main__":
    main()
