#!/usr/bin/env python
"""Render h1_reference_motion.npz files into H1 reference GIFs.

This visualizes robot-level reference motion before policy/simulator execution.
It reads the pipeline-native H1 reference format directly instead of converting
through FRoM-W1 pkl.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H1_XML = (
    REPO_ROOT.parent / "FRoM-W1" / "H-ACT" / "retarget" / "assets" / "robot" / "h1" / "h1.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render h1_reference_motion.npz to H1 GIF.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*_h1_reference_motion.npz")
    parser.add_argument("--output-gif", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/gifs/h1_reference_gifs"))
    parser.add_argument("--xml", type=Path, default=DEFAULT_H1_XML)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--ambient", type=float, default=0.8)
    parser.add_argument("--diffuse", type=float, default=1.0)
    parser.add_argument("--specular", type=float, default=0.3)
    parser.add_argument("--exposure", type=float, default=2.2)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--camera-azimuth", type=float, default=180.0)
    parser.add_argument("--camera-elevation", type=float, default=-6.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.4)
    parser.add_argument("--camera-min-distance", type=float, default=3.4)
    parser.add_argument("--camera-lookat-height", type=float, default=0.72)
    parser.add_argument("--no-auto-fit", action="store_true")
    parser.add_argument("--auto-fit-margin", type=float, default=0.22)
    return parser.parse_args()


def clean_stem(path: Path) -> str:
    stem = path.stem
    suffix = "_h1_reference_motion"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def iter_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_npz:
        return [args.input_npz.resolve()]
    return sorted(p.resolve() for p in args.input_dir.glob(args.glob) if p.is_file())


def require_fields(data: np.lib.npyio.NpzFile, path: Path) -> None:
    required = ["root_pos_ref", "root_quat_ref_xyzw", "h1_dof_pos_ref", "fps"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")


def load_h1_reference(path: Path) -> dict[str, np.ndarray | int]:
    data = np.load(str(path), allow_pickle=False)
    require_fields(data, path)
    root_pos = data["root_pos_ref"].astype(np.float32)
    root_quat_xyzw = data["root_quat_ref_xyzw"].astype(np.float32)
    dof = data["h1_dof_pos_ref"].astype(np.float32)
    fps = int(data["fps"])

    if dof.ndim != 2:
        raise ValueError(f"{path} h1_dof_pos_ref must be 2D, got {dof.shape}")
    if root_pos.shape != (dof.shape[0], 3):
        raise ValueError(f"{path} root_pos_ref shape {root_pos.shape}, expected {(dof.shape[0], 3)}")
    if root_quat_xyzw.shape != (dof.shape[0], 4):
        raise ValueError(f"{path} root_quat_ref_xyzw shape {root_quat_xyzw.shape}, expected {(dof.shape[0], 4)}")

    return {
        "root_pos": root_pos,
        "root_quat_xyzw": root_quat_xyzw,
        "dof": dof,
        "fps": fps,
    }


def make_camera(
    root_positions: np.ndarray,
    azimuth: float,
    elevation: float,
    distance_scale: float,
    min_distance: float,
    lookat_height: float,
) -> mujoco.MjvCamera:
    mins = root_positions.min(axis=0)
    maxs = root_positions.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.linalg.norm(maxs[[0, 2]] - mins[[0, 2]]))

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([center[0], lookat_height, center[2]])
    cam.distance = max(min_distance, 2.2 + span * distance_scale)
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def configure_lighting(model: mujoco.MjModel, ambient: float, diffuse: float, specular: float) -> None:
    model.vis.headlight.ambient[:] = np.full(3, ambient, dtype=np.float32)
    model.vis.headlight.diffuse[:] = np.full(3, diffuse, dtype=np.float32)
    model.vis.headlight.specular[:] = np.full(3, specular, dtype=np.float32)


def adjust_image_brightness(frame: np.ndarray, exposure: float, gamma: float) -> np.ndarray:
    image = np.asarray(frame, dtype=np.float32) / 255.0
    image = np.clip(image * exposure, 0.0, 1.0)
    if gamma != 1.0:
        image = np.power(image, gamma)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def center_subject(frame: np.ndarray, threshold: int = 12) -> np.ndarray:
    mask = np.max(frame, axis=2) > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return frame

    height, width = frame.shape[:2]
    cx = (int(xs.min()) + int(xs.max())) // 2
    cy = (int(ys.min()) + int(ys.max())) // 2
    shift_x = width // 2 - cx
    shift_y = height // 2 - cy

    output = np.zeros_like(frame)
    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x) if shift_x >= 0 else width
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y) if shift_y >= 0 else height
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    output[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return output


def fit_subject_to_frame(frame: np.ndarray, margin_ratio: float, threshold: int = 12) -> np.ndarray:
    mask = np.max(frame, axis=2) > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return frame

    height, width = frame.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    subject_w = max(1, x1 - x0)
    subject_h = max(1, y1 - y0)
    margin = max(0.0, float(margin_ratio))

    crop_w = int(round(subject_w * (1.0 + margin * 2.0)))
    crop_h = int(round(subject_h * (1.0 + margin * 2.0)))
    target_aspect = width / height
    crop_aspect = crop_w / max(1, crop_h)
    if crop_aspect < target_aspect:
        crop_w = int(round(crop_h * target_aspect))
    else:
        crop_h = int(round(crop_w / target_aspect))

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    left = max(0, cx - crop_w // 2)
    top = max(0, cy - crop_h // 2)
    right = min(width, left + crop_w)
    bottom = min(height, top + crop_h)
    left = max(0, right - crop_w)
    top = max(0, bottom - crop_h)

    crop = frame[top:bottom, left:right]
    image = Image.fromarray(crop).resize((width, height), Image.Resampling.LANCZOS)
    return center_subject(np.asarray(image, dtype=np.uint8), threshold=threshold)


def frame_to_qpos(ref: dict[str, np.ndarray | int], frame_idx: int, model_nq: int) -> np.ndarray:
    qpos = np.zeros(model_nq, dtype=np.float64)
    qpos[:3] = ref["root_pos"][frame_idx]
    quat_xyzw = np.asarray(ref["root_quat_xyzw"][frame_idx], dtype=np.float64)
    qpos[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    qpos[7:] = np.asarray(ref["dof"][frame_idx], dtype=np.float64)
    return qpos


def render_one(
    input_npz: Path,
    output_gif: Path,
    xml_path: Path,
    width: int,
    height: int,
    output_fps: int | None,
    step: int,
    ambient: float,
    diffuse: float,
    specular: float,
    exposure: float,
    gamma: float,
    camera_azimuth: float,
    camera_elevation: float,
    camera_distance_scale: float,
    camera_min_distance: float,
    camera_lookat_height: float,
    auto_fit: bool,
    auto_fit_margin: float,
) -> None:
    ref = load_h1_reference(input_npz)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    configure_lighting(model, ambient=ambient, diffuse=diffuse, specular=specular)

    dof = ref["dof"]
    if dof.shape[1] != model.nq - 7:
        raise ValueError(f"{input_npz} has {dof.shape[1]} DOF, but XML expects {model.nq - 7}.")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = make_camera(
        np.asarray(ref["root_pos"]),
        azimuth=camera_azimuth,
        elevation=camera_elevation,
        distance_scale=camera_distance_scale,
        min_distance=camera_min_distance,
        lookat_height=camera_lookat_height,
    )

    frames: list[Image.Image] = []
    for frame_idx in range(0, dof.shape[0], max(1, step)):
        data.qpos[:] = frame_to_qpos(ref, frame_idx, model.nq)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = adjust_image_brightness(renderer.render(), exposure=exposure, gamma=gamma)
        if auto_fit:
            frame = fit_subject_to_frame(frame, margin_ratio=auto_fit_margin)
        frames.append(Image.fromarray(frame))

    if not frames:
        raise ValueError(f"{input_npz} produced no frames.")

    save_fps = int(output_fps or max(1, int(ref["fps"]) // max(1, step)))
    duration_ms = max(1, int(round(1000.0 / save_fps)))
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    renderer.close()
    print(f"[DONE] {input_npz} -> {output_gif} frames={len(frames)} fps={save_fps}")


def main() -> None:
    args = parse_args()
    inputs = iter_inputs(args)
    if not inputs:
        raise FileNotFoundError("No h1_reference_motion.npz files matched.")
    if args.output_gif and len(inputs) != 1:
        raise ValueError("--output-gif can only be used with one input.")

    xml_path = args.xml.resolve()
    for input_npz in inputs:
        output_gif = args.output_gif or args.output_dir / f"{clean_stem(input_npz)}_h1_reference.gif"
        render_one(
            input_npz=input_npz,
            output_gif=output_gif.resolve(),
            xml_path=xml_path,
            width=args.width,
            height=args.height,
            output_fps=args.fps,
            step=max(1, args.step),
            ambient=args.ambient,
            diffuse=args.diffuse,
            specular=args.specular,
            exposure=args.exposure,
            gamma=args.gamma,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            camera_distance_scale=args.camera_distance_scale,
            camera_min_distance=args.camera_min_distance,
            camera_lookat_height=args.camera_lookat_height,
            auto_fit=not args.no_auto_fit,
            auto_fit_margin=args.auto_fit_margin,
        )


if __name__ == "__main__":
    main()
