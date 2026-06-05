#!/usr/bin/env python
"""Convert canonical_motion.npz to FRoM-W1 normalized 623 feature npy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.smpl_npy_to_fromw1_623 import _joints52_to_623, _stabilize_lower_body  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert canonical_motion.npz to FRoM-W1 623 npy.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*_canonical_motion.npz")
    parser.add_argument("--output-npy", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fromw1_inputs"))
    parser.add_argument("--fromw1-root", type=Path, default=REPO_ROOT.parent / "FRoM-W1")
    parser.add_argument("--reference-623", type=Path, default=None)
    parser.add_argument("--feet-thre", type=float, default=0.002)
    parser.add_argument("--no-stabilize-lower-body", action="store_true")
    return parser.parse_args()


def clean_stem(path: Path) -> str:
    stem = path.stem
    suffix = "_canonical_motion"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def resolve_reference_623(fromw1_root: Path, reference_623: Path | None) -> Path:
    if reference_623 is not None:
        return reference_623.resolve()
    return (fromw1_root / "H-ACT" / "retarget" / "data" / "623" / "0_feats_out.npy").resolve()


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_npz:
        return [args.input_npz.resolve()]
    return sorted(p.resolve() for p in args.input_dir.glob(args.glob) if p.is_file())


def convert_one(
    input_npz: Path,
    output_npy: Path,
    fromw1_root: Path,
    reference_623: Path,
    feet_thre: float,
    stabilize_lower_body: bool,
) -> np.ndarray:
    data = np.load(str(input_npz), allow_pickle=False)
    if "motionx52_joints" not in data:
        raise ValueError(f"{input_npz} does not contain motionx52_joints")
    joints52 = data["motionx52_joints"].astype(np.float32)

    if stabilize_lower_body:
        joints52 = _stabilize_lower_body(
            joints52=joints52,
            reference_623=reference_623,
            fromw1_root=fromw1_root,
        )

    features_623 = _joints52_to_623(
        joints52=joints52,
        fromw1_root=fromw1_root,
        feet_thre=feet_thre,
        reference_623=reference_623,
    )
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_npy), features_623)
    return features_623


def main() -> None:
    args = parse_args()
    fromw1_root = args.fromw1_root.resolve()
    reference_623 = resolve_reference_623(fromw1_root, args.reference_623)
    inputs = iter_inputs(args)
    if not inputs:
        raise FileNotFoundError("No canonical npz files matched.")
    if args.output_npy and len(inputs) != 1:
        raise ValueError("--output-npy can only be used with one input.")

    for input_npz in inputs:
        output_npy = args.output_npy or args.output_dir / f"{clean_stem(input_npz)}_623.npy"
        features = convert_one(
            input_npz=input_npz,
            output_npy=output_npy.resolve(),
            fromw1_root=fromw1_root,
            reference_623=reference_623,
            feet_thre=args.feet_thre,
            stabilize_lower_body=not args.no_stabilize_lower_body,
        )
        print(f"[DONE] {input_npz} -> {output_npy.resolve()} shape={features.shape} dtype={features.dtype}")


if __name__ == "__main__":
    main()
