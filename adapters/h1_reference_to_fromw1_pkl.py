#!/usr/bin/env python
"""Convert h1_reference_motion.npz back to a FRoM-W1/RoboJuDo-compatible pkl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert h1_reference_motion.npz to FRoM-W1 H1 pkl.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*_h1_reference_motion.npz")
    parser.add_argument("--output-pkl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fromw1_pkl/fromw1_pkl_h1_reference"))
    parser.add_argument("--motion-key", default="motion 0")
    return parser.parse_args()


def clean_stem(path: Path) -> str:
    stem = path.stem
    suffix = "_h1_reference_motion"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_npz:
        return [args.input_npz.resolve()]
    return sorted(p.resolve() for p in args.input_dir.glob(args.glob) if p.is_file())


def get_optional(data: np.lib.npyio.NpzFile, key: str, default):
    return data[key] if key in data.files else default


def convert_one(input_npz: Path, output_pkl: Path, motion_key: str) -> None:
    data = np.load(str(input_npz), allow_pickle=False)
    required = ["root_pos_ref", "root_quat_ref_xyzw", "h1_dof_pos_ref", "fps"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{input_npz} missing required keys: {missing}")

    root_pos = data["root_pos_ref"].astype(np.float32)
    root_quat = data["root_quat_ref_xyzw"].astype(np.float32)
    dof = data["h1_dof_pos_ref"].astype(np.float32)
    fps = int(data["fps"])

    if root_pos.shape != (dof.shape[0], 3):
        raise ValueError(f"root_pos_ref shape {root_pos.shape} does not match dof frames {dof.shape[0]}")
    if root_quat.shape != (dof.shape[0], 4):
        raise ValueError(f"root_quat_ref_xyzw shape {root_quat.shape} does not match dof frames {dof.shape[0]}")

    pose_aa = get_optional(data, "h1_body_pose_aa_ref", np.zeros((dof.shape[0], 0, 3), dtype=np.float32)).astype(
        np.float32
    )
    hand_pose = get_optional(data, "hand_pose_ref", np.zeros((dof.shape[0], 0), dtype=np.float32)).astype(np.float32)
    body_names = get_optional(data, "h1_body_names", np.asarray([], dtype=str)).astype(str).tolist()

    motion = {
        "body_names": body_names,
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat,
        "fps": fps,
        "hand_pose": hand_pose,
    }

    metadata = {}
    if "metadata_json" in data.files:
        try:
            metadata = json.loads(str(data["metadata_json"]))
        except json.JSONDecodeError:
            metadata = {}
    if metadata:
        motion["h1_reference_metadata"] = metadata

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion_key: motion}, output_pkl)
    print(f"[DONE] {input_npz} -> {output_pkl} frames={dof.shape[0]} fps={fps}")


def main() -> None:
    args = parse_args()
    inputs = iter_inputs(args)
    if not inputs:
        raise FileNotFoundError("No h1_reference npz files matched.")
    if args.output_pkl and len(inputs) != 1:
        raise ValueError("--output-pkl can only be used with one input.")

    for input_npz in inputs:
        output_pkl = args.output_pkl or args.output_dir / f"{clean_stem(input_npz)}.pkl"
        convert_one(input_npz=input_npz, output_pkl=output_pkl.resolve(), motion_key=args.motion_key)


if __name__ == "__main__":
    main()
