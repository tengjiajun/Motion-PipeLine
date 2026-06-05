import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from PIL import Image


SMPL24_CHAINS = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20, 22],
    [9, 14, 17, 19, 21, 23],
]

CHAIN_COLORS = ["red", "blue", "black", "red", "blue"]
CHAIN_WIDTHS = [4.0, 4.0, 4.0, 4.0, 4.0]


def pick_person(frame_result):
    if not frame_result:
        return None
    return max(frame_result, key=lambda item: float(item.get("bbox_score", 0.0)))


def forward_fill(frames):
    valid_indices = [i for i, pose in enumerate(frames) if pose is not None]
    if not valid_indices:
        raise ValueError("No valid poses were found in the input file.")

    first_valid = valid_indices[0]
    last_pose = frames[first_valid]

    for i in range(first_valid):
        frames[i] = last_pose.copy()

    for i in range(first_valid, len(frames)):
        if frames[i] is None:
            frames[i] = last_pose.copy()
        else:
            last_pose = frames[i]

    return np.stack(frames, axis=0)


def load_motion(npy_path: Path, joint_key: str):
    payload = np.load(npy_path, allow_pickle=True).item()
    frame_entries = payload["frames"]
    motion = []
    for frame in frame_entries:
        person = pick_person(frame["result"])
        if person is None:
            motion.append(None)
            continue
        motion.append(np.asarray(person[joint_key], dtype=np.float32))
    return forward_fill(motion)


def make_floor(ax, minx, maxx, miny, minz, maxz):
    verts = [
        [minx, miny, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, miny, minz],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(plane)


def render_gif(motion, output_path: Path, fps: float):
    data = motion.copy().reshape(len(motion), -1, 3)
    # AlphaPose joints use a camera-centric convention where -Y is "up".
    # Convert to an upright plotting convention: X right, Y up, Z depth.
    data = np.stack([data[..., 0], -data[..., 1], data[..., 2]], axis=-1)
    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)

    data[:, :, 1] -= mins[1]
    traj = data[:, 0, [0, 2]].copy()
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    radius = max(
        float(maxs[0] - mins[0]),
        float(maxs[2] - mins[2]),
        1.5,
    )
    height = max(float(data[:, :, 1].max()), 1.5)

    fig = plt.figure(figsize=(9.6, 9.6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")

    def update(index):
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        make_floor(
            ax,
            mins[0] - traj[index, 0],
            maxs[0] - traj[index, 0],
            0.0,
            mins[2] - traj[index, 1],
            maxs[2] - traj[index, 1],
        )

        for chain, color, linewidth in zip(SMPL24_CHAINS, CHAIN_COLORS, CHAIN_WIDTHS):
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        ax.set_xlim3d([-radius / 2.0, radius / 2.0])
        ax.set_ylim3d([0.0, max(height, radius * 0.8)])
        ax.set_zlim3d([-radius / 2.0, radius / 2.0])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)
        plt.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx in range(data.shape[0]):
        update(idx)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(frame))

    duration_ms = max(1, int(round(1000.0 / fps)))
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render AlphaPose SMPL npy to a skeleton GIF.")
    parser.add_argument("--input", required=True, help="Input AlphaPose SMPL npy.")
    parser.add_argument("--output", default="", help="Output GIF path.")
    parser.add_argument(
        "--joint-key",
        default="pred_xyz_jts_24",
        choices=["pred_xyz_jts_24", "pred_xyz_jts_24_struct", "pred_xyz_jts_29"],
        help="Which joint array from the npy payload to render.",
    )
    parser.add_argument("--fps", type=float, default=29.0, help="Output GIF fps.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_skeleton.gif").resolve()
    )

    motion = load_motion(input_path, args.joint_key)
    render_gif(motion, output_path, fps=args.fps)
    print(output_path)


if __name__ == "__main__":
    main()
