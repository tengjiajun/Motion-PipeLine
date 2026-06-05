#!/usr/bin/env python
"""Batch retarget FRoM-W1 623 feature npy files into H1 pkl motions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FRoM-W1 623 npy files to H1 pkl files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*_623.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smplx-dir", type=Path, default=None)
    parser.add_argument("--fromw1-retarget-root", type=Path, default=Path(r"F:\LLM-pepper\FRoM-W1\H-ACT\retarget"))
    parser.add_argument("--robot", default="H1", choices=["H1", "G1", "H121"])
    parser.add_argument("--hand-type", default="dex3", choices=["inspire", "dex3"])
    parser.add_argument("--output-fps", type=int, default=60)
    return parser.parse_args()


def import_fromw1(retarget_root: Path):
    retarget_root = retarget_root.resolve()
    if str(retarget_root) not in sys.path:
        sys.path.insert(0, str(retarget_root))
    from body_retarget import load_amass_data, process_data  # noqa: WPS433
    from hand_retarget import retarget_from_rotvec  # noqa: WPS433
    from utils import feats2joints, pos2smpl, set_fps  # noqa: WPS433

    return load_amass_data, process_data, retarget_from_rotvec, feats2joints, pos2smpl, set_fps


def clean_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_canonical_motion_v2_623", "_canonical_motion_623", "_623"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_feature_tensor(path: Path) -> torch.Tensor:
    arr = np.load(str(path), allow_pickle=True)
    if not isinstance(arr, np.ndarray):
        raise ValueError("feature file is not a numpy array")
    if arr.dtype == object:
        raise ValueError("feature file contains object payload, not 623D features")
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[-1] != 623:
        raise ValueError(f"expected shape (1, T, 623), got {arr.shape}")
    return torch.from_numpy(arr[0])


def convert_one(
    input_npy: Path,
    output_pkl: Path,
    smplx_npz: Path,
    funcs,
    robot: str,
    hand_type: str,
    output_fps: int,
) -> None:
    load_amass_data, process_data, retarget_from_rotvec, feats2joints, pos2smpl, set_fps = funcs
    data = load_feature_tensor(input_npy)
    smplx_pos = feats2joints(data)
    smpl_dict = pos2smpl(smplx_pos)

    smplx_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(smplx_npz), **smpl_dict)

    amass_data = load_amass_data(str(smplx_npz))
    robot_data = process_data(amass_data, robot)
    hand_data = retarget_from_rotvec(smpl_dict["poses"][:, 66:], hand_type=hand_type)
    robot_data.update({"hand_pose": hand_data})
    robot_data = {"motion 0": set_fps(robot_data, output_fps)}

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(robot_data, str(output_pkl))
    print(f"[DONE] {input_npy} -> {output_pkl} dof={robot_data['motion 0']['dof'].shape}")


def main() -> None:
    args = parse_args()
    input_paths = sorted(p for p in args.input_dir.glob(args.glob) if p.is_file())
    if not input_paths:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.glob}")

    smplx_dir = args.smplx_dir or args.output_dir / "smplx"
    funcs = import_fromw1(args.fromw1_retarget_root)
    for input_npy in input_paths:
        stem = clean_stem(input_npy)
        output_pkl = args.output_dir / f"{stem}_623.pkl"
        smplx_npz = smplx_dir / f"{stem}_623.npz"
        convert_one(
            input_npy=input_npy.resolve(),
            output_pkl=output_pkl.resolve(),
            smplx_npz=smplx_npz.resolve(),
            funcs=funcs,
            robot=args.robot,
            hand_type=args.hand_type,
            output_fps=args.output_fps,
        )


if __name__ == "__main__":
    main()
