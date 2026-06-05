#!/usr/bin/env python
"""Batch repair AlphaPose SMPL raw npy files.

Outputs:
  - data/smpl_repaired/<motion_id>_smpl_repaired.npy
  - data/smpl_repaired_compact/<motion_id>_smpl_repaired_compact.npy
  - data/metrics/alphapose_repair/repair_summary.{csv,json}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.repair_smpl_npy import repair_track  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch repair AlphaPose SMPL raw npy files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/smpl_raw"))
    parser.add_argument("--repaired-dir", type=Path, default=Path("data/smpl_repaired"))
    parser.add_argument("--compact-dir", type=Path, default=Path("data/smpl_repaired_compact"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("data/metrics/alphapose_repair"))
    parser.add_argument("--pattern", default="*_smpl_raw.npy")
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--spike-z", type=float, default=4.0)
    parser.add_argument("--bridge-scale", type=float, default=1.5)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--frame-bad-ratio", type=float, default=0.35)
    return parser.parse_args()


def motion_id_from_path(path: Path) -> str:
    suffix = "_smpl_raw"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def load_payload(path: Path) -> dict[str, Any]:
    obj = np.load(str(path), allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()
    raise ValueError(f"Unsupported npy structure: {path}")


def frame_has_person(frame: dict[str, Any]) -> bool:
    return bool(frame.get("result"))


def compact_by_raw_valid_frames(repaired: dict[str, Any], raw: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    keep_indices = [i for i, frame in enumerate(raw["frames"]) if frame_has_person(frame)]
    if not keep_indices:
        raise ValueError(f"No valid frames to keep in {raw_path}")
    if len(repaired["frames"]) != len(raw["frames"]):
        raise ValueError(f"Raw/repaired frame count mismatch for {raw_path}")

    compact = dict(repaired)
    compact["frames"] = [repaired["frames"][i] for i in keep_indices]
    compact["compact_meta"] = {
        "source": str(raw_path),
        "source_frames": len(raw["frames"]),
        "kept_frames": len(keep_indices),
        "dropped_frames": len(raw["frames"]) - len(keep_indices),
        "rule": "drop frames where original smpl_raw frame has no detected person",
    }
    return compact


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


def write_summary(rows: list[dict[str, Any]], metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "motion_id",
        "source_file",
        "repaired_file",
        "compact_file",
        "source_frames",
        "compact_frames",
        "dropped_frames",
        "missing_frames",
        "joint24_repaired",
        "joint29_repaired",
        "transl_repaired",
        "cam_root_repaired",
        "smooth_window",
        "min_conf",
        "spike_z",
        "bridge_scale",
        "frame_bad_ratio",
    ]
    with (metrics_dir / "repair_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (metrics_dir / "repair_summary.json").write_text(
        json.dumps(to_jsonable(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    paths = sorted(p for p in args.input_dir.glob(args.pattern) if p.is_file() and p.stat().st_size > 0)
    if not paths:
        raise SystemExit(f"No files matched {args.input_dir / args.pattern}")

    args.repaired_dir.mkdir(parents=True, exist_ok=True)
    args.compact_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw_path in paths:
        motion_id = motion_id_from_path(raw_path)
        raw = load_payload(raw_path)
        repaired = repair_track(
            payload=raw,
            min_conf=args.min_conf,
            spike_z=args.spike_z,
            bridge_scale=args.bridge_scale,
            smooth_window=args.smooth_window,
            frame_bad_ratio=args.frame_bad_ratio,
        )
        compact = compact_by_raw_valid_frames(repaired, raw, raw_path)

        repaired_path = args.repaired_dir / f"{motion_id}_smpl_repaired.npy"
        compact_path = args.compact_dir / f"{motion_id}_smpl_repaired_compact.npy"
        np.save(str(repaired_path), repaired, allow_pickle=True)
        np.save(str(compact_path), compact, allow_pickle=True)

        meta = repaired["repair_meta"]
        compact_meta = compact["compact_meta"]
        root_meta = meta.get("root_repaired", {})
        rows.append(
            {
                "motion_id": motion_id,
                "source_file": str(raw_path),
                "repaired_file": str(repaired_path),
                "compact_file": str(compact_path),
                "source_frames": compact_meta["source_frames"],
                "compact_frames": compact_meta["kept_frames"],
                "dropped_frames": compact_meta["dropped_frames"],
                "missing_frames": meta["missing_frames"],
                "joint24_repaired": meta["joint24_repaired"],
                "joint29_repaired": meta["joint29_repaired"],
                "transl_repaired": root_meta.get("transl", 0),
                "cam_root_repaired": root_meta.get("cam_root", 0),
                "smooth_window": meta["smooth_window"],
                "min_conf": meta["min_conf"],
                "spike_z": meta["spike_z"],
                "bridge_scale": meta["bridge_scale"],
                "frame_bad_ratio": meta["frame_bad_ratio"],
            }
        )
        print(f"{motion_id}: {repaired_path} -> {compact_path}")

    write_summary(rows, args.metrics_dir)
    print(f"Wrote repair summary to {args.metrics_dir}")


if __name__ == "__main__":
    main()
