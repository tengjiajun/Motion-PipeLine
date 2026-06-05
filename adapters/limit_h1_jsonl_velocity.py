#!/usr/bin/env python
"""Apply joint-limit and velocity limiting to an H1 upper-body JSONL motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


H1_UPPER_LIMITS: dict[str, tuple[float, float]] = {
    "torso": (-2.35, 2.35),
    "left_shoulder_pitch": (-2.87, 2.87),
    "left_shoulder_roll": (-0.34, 3.11),
    "left_shoulder_yaw": (-1.3, 4.45),
    "left_elbow": (-1.25, 2.61),
    "right_shoulder_pitch": (-2.87, 2.87),
    "right_shoulder_roll": (-3.11, 0.34),
    "right_shoulder_yaw": (-4.45, 1.3),
    "right_elbow": (-1.25, 2.61),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    if not frames:
        raise ValueError(f"Empty JSONL: {path}")
    return frames


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def limit_motion(
    frames: list[dict[str, Any]],
    max_vel_rad_s: float,
    *,
    clip_limits: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_vel_rad_s <= 0.0:
        raise ValueError("max_vel_rad_s must be positive")

    out_frames: list[dict[str, Any]] = []
    prev_angles: dict[str, float] | None = None
    prev_t: float | None = None

    clamp_count = 0
    velocity_limit_count = 0
    stats: dict[str, dict[str, Any]] = {}

    for frame_idx, frame in enumerate(frames):
        names = list(frame["joint_names"])
        raw_angles = [float(v) for v in frame["angles"]]
        t = float(frame.get("video_timestamp", frame_idx))

        limited_angles: list[float] = []
        for name, raw_value in zip(names, raw_angles, strict=True):
            value = raw_value
            limits = H1_UPPER_LIMITS.get(name)

            if clip_limits and limits is not None:
                clipped = _clamp(value, limits[0], limits[1])
                if abs(clipped - value) > 1e-8:
                    clamp_count += 1
                value = clipped

            if prev_angles is not None and prev_t is not None and name in prev_angles:
                dt = max(1e-6, t - prev_t)
                max_step = max_vel_rad_s * dt
                prev_value = prev_angles[name]
                delta = value - prev_value
                if delta > max_step:
                    value = prev_value + max_step
                    velocity_limit_count += 1
                elif delta < -max_step:
                    value = prev_value - max_step
                    velocity_limit_count += 1

                if clip_limits and limits is not None:
                    value = _clamp(value, limits[0], limits[1])

            limited_angles.append(value)

            joint_stats = stats.setdefault(
                name,
                {
                    "min": value,
                    "max": value,
                    "max_step": 0.0,
                    "max_velocity": 0.0,
                    "limit": list(limits) if limits is not None else None,
                },
            )
            joint_stats["min"] = min(float(joint_stats["min"]), value)
            joint_stats["max"] = max(float(joint_stats["max"]), value)
            if prev_angles is not None and prev_t is not None and name in prev_angles:
                dt = max(1e-6, t - prev_t)
                step = abs(value - prev_angles[name])
                joint_stats["max_step"] = max(float(joint_stats["max_step"]), step)
                joint_stats["max_velocity"] = max(float(joint_stats["max_velocity"]), step / dt)

        out_frame = dict(frame)
        out_frame["angles"] = limited_angles
        out_frame["source"] = "h1_jsonl_velocity_limited"
        out_frame["max_vel_rad_s"] = max_vel_rad_s
        out_frames.append(out_frame)

        prev_angles = dict(zip(names, limited_angles, strict=True))
        prev_t = t

    meta = {
        "frames": len(out_frames),
        "duration_s": float(frames[-1].get("video_timestamp", len(frames) - 1))
        - float(frames[0].get("video_timestamp", 0.0)),
        "max_vel_rad_s": max_vel_rad_s,
        "clip_limits": clip_limits,
        "clamp_count": clamp_count,
        "velocity_limit_count": velocity_limit_count,
        "stats": stats,
    }
    return out_frames, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument("--max-vel-rad-s", type=float, default=4.5)
    parser.add_argument("--no-clip-limits", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(input_path.stem + "_vel_limited.jsonl")
    )
    meta_path = (
        args.meta_output.resolve()
        if args.meta_output is not None
        else output_path.with_suffix(".meta.json")
    )

    frames = _load_jsonl(input_path)
    out_frames, meta = limit_motion(
        frames,
        max_vel_rad_s=float(args.max_vel_rad_s),
        clip_limits=not bool(args.no_clip_limits),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for frame in out_frames:
            f.write(json.dumps(frame, ensure_ascii=False) + "\n")

    meta["input"] = str(input_path)
    meta["output"] = str(output_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
