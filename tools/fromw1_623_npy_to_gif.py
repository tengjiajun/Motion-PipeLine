import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


BODY_CHAINS = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

LEFT_HAND_CHAINS = [
    [20, 22, 23, 24],
    [20, 34, 35, 36],
    [20, 25, 26, 27],
    [20, 31, 32, 33],
    [20, 28, 29, 30],
]

RIGHT_HAND_CHAINS = [
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 40, 41, 42],
    [21, 37, 38, 39],
    [21, 49, 50, 51],
]


def _load_features(path: Path) -> np.ndarray:
    data = np.load(str(path), allow_pickle=True)
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError(f"{path} must have batch size 1, got {data.shape}")
        data = data[0]
    if data.ndim != 2 or data.shape[-1] != 623:
        raise ValueError(f"{path} must be (T, 623) or (1, T, 623), got {data.shape}")
    return np.asarray(data, dtype=np.float32)


def _recover_joints(features: np.ndarray, fromw1_root: Path) -> np.ndarray:
    retarget_root = (fromw1_root / "H-ACT" / "retarget").resolve()
    if str(retarget_root) not in sys.path:
        sys.path.insert(0, str(retarget_root))

    old_cwd = Path.cwd()
    os.chdir(retarget_root)
    try:
        from utils import feats2joints  # noqa: WPS433

        with torch.no_grad():
            joints = feats2joints(torch.from_numpy(features).float())
        return joints.detach().cpu().numpy().astype(np.float32)
    finally:
        os.chdir(old_cwd)


def _project_xy(joints: np.ndarray, width: int, height: int, pad: int) -> np.ndarray:
    x = joints[..., 0]
    y = joints[..., 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    scale_x = (width - 2 * pad) / max(1e-6, xmax - xmin)
    scale_y = (height - 2 * pad) / max(1e-6, ymax - ymin)
    scale = min(scale_x, scale_y) * 0.92

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    points = np.zeros(joints.shape[:2] + (2,), dtype=np.float32)
    points[..., 0] = (x - cx) * scale + width * 0.5
    points[..., 1] = height * 0.5 - (y - cy) * scale
    return points


def _project_camera(
    joints: np.ndarray,
    width: int,
    height: int,
    pad: int,
    azim_deg: float,
    elev_deg: float,
) -> np.ndarray:
    azim = np.deg2rad(azim_deg)
    elev = np.deg2rad(elev_deg)
    x = joints[..., 0]
    y = joints[..., 1]
    z = joints[..., 2]

    view_x = x * np.cos(azim) - z * np.sin(azim)
    view_z = x * np.sin(azim) + z * np.cos(azim)
    view_y = y * np.cos(elev) - view_z * np.sin(elev)

    xmin, xmax = float(np.min(view_x)), float(np.max(view_x))
    ymin, ymax = float(np.min(view_y)), float(np.max(view_y))
    scale_x = (width - 2 * pad) / max(1e-6, xmax - xmin)
    scale_y = (height - 2 * pad) / max(1e-6, ymax - ymin)
    scale = min(scale_x, scale_y) * 0.92

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    points = np.zeros(joints.shape[:2] + (2,), dtype=np.float32)
    points[..., 0] = (view_x - cx) * scale + width * 0.5
    points[..., 1] = height * 0.5 - (view_y - cy) * scale
    return points


def _draw_chain(draw: ImageDraw.ImageDraw, points: np.ndarray, chain: list[int], color, width: int):
    for start, end in zip(chain[:-1], chain[1:]):
        draw.line((tuple(points[start]), tuple(points[end])), fill=color, width=width)


def _draw_frame(points: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (244, 244, 242))
    draw = ImageDraw.Draw(image)

    for chain in BODY_CHAINS:
        _draw_chain(draw, points, chain, (20, 20, 20), 5)
    for chain in LEFT_HAND_CHAINS:
        _draw_chain(draw, points, chain, (35, 85, 220), 3)
    for chain in RIGHT_HAND_CHAINS:
        _draw_chain(draw, points, chain, (220, 45, 45), 3)

    for idx, point in enumerate(points):
        radius = 4 if idx < 22 else 2
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(25, 25, 25))
    return image


def render_gif(
    input_npy: Path,
    output_gif: Path,
    fromw1_root: Path,
    width: int,
    height: int,
    fps: int,
    step: int,
    projection: str,
    azim: float,
    elev: float,
) -> tuple[int, tuple[int, int]]:
    features = _load_features(input_npy)
    joints = _recover_joints(features, fromw1_root=fromw1_root)
    if projection == "xy":
        points = _project_xy(joints, width=width, height=height, pad=64)
    else:
        points = _project_camera(
            joints,
            width=width,
            height=height,
            pad=64,
            azim_deg=azim,
            elev_deg=elev,
        )

    frames = [
        _draw_frame(points[i], width=width, height=height)
        for i in range(0, len(points), max(1, step))
    ]
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(1000 / max(1, fps)))),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return len(frames), (width, height)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render FRoM-W1 normalized 623 npy files to GIF.")
    parser.add_argument("--input", type=Path, help="Single .npy input.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing .npy inputs.")
    parser.add_argument("--output", type=Path, help="Single .gif output.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "gifs" / "fromw1623npy")
    parser.add_argument("--fromw1-root", type=Path, default=Path("../FRoM-W1"))
    parser.add_argument("--width", type=int, default=760)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--projection",
        choices=["camera", "xy"],
        default="camera",
        help="camera keeps depth; xy is the old front projection.",
    )
    parser.add_argument("--azim", type=float, default=-45.0, help="Camera azimuth for --projection camera.")
    parser.add_argument("--elev", type=float, default=12.0, help="Camera elevation for --projection camera.")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.input) == bool(args.input_dir):
        raise SystemExit("Set exactly one of --input or --input-dir.")

    fromw1_root = args.fromw1_root.resolve()
    if args.input:
        output = args.output or args.output_dir / f"{args.input.stem}.gif"
        count, size = render_gif(
            args.input.resolve(),
            output.resolve(),
            fromw1_root=fromw1_root,
            width=args.width,
            height=args.height,
            fps=args.fps,
            step=args.step,
            projection=args.projection,
            azim=args.azim,
            elev=args.elev,
        )
        print(f"[DONE] {args.input} -> {output} frames={count} size={size}")
        return

    output_dir = args.output_dir.resolve()
    for input_npy in sorted(args.input_dir.resolve().glob("*.npy")):
        output = output_dir / f"{input_npy.stem}.gif"
        count, size = render_gif(
            input_npy,
            output,
            fromw1_root=fromw1_root,
            width=args.width,
            height=args.height,
            fps=args.fps,
            step=args.step,
            projection=args.projection,
            azim=args.azim,
            elev=args.elev,
        )
        print(f"[DONE] {input_npy.name} -> {output.name} frames={count} size={size}")


if __name__ == "__main__":
    main()
