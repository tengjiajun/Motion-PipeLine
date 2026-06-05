import argparse
from pathlib import Path

import joblib
import numpy as np


def _load_motion(pkl_path: Path):
    data = joblib.load(pkl_path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Invalid pkl structure: {pkl_path}")
    key = next(iter(data.keys()))
    motion = data[key]
    if "pose_aa" not in motion or "root_trans_offset" not in motion:
        raise ValueError(f"Missing required keys in {pkl_path}: pose_aa/root_trans_offset")
    return data, key, motion


def _interp_time(arr: np.ndarray, target_len: int) -> np.ndarray:
    src_len = arr.shape[0]
    if src_len == target_len:
        return arr.astype(np.float32, copy=True)

    t_src = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    t_tar = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    flat = arr.reshape(src_len, -1)
    out = np.vstack([np.interp(t_tar, t_src, flat[:, i]) for i in range(flat.shape[1])]).T
    return out.reshape((target_len,) + arr.shape[1:]).astype(np.float32)


def _smooth_2d(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr.astype(np.float32, copy=True)
    pad = win // 2
    ker = np.ones(win, dtype=np.float32) / win
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[1]):
        out[:, c] = np.convolve(padded[:, c], ker, mode="valid")
    return out


def _smooth_3d(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr.astype(np.float32, copy=True)
    pad = win // 2
    ker = np.ones(win, dtype=np.float32) / win
    padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for j in range(arr.shape[1]):
        for k in range(arr.shape[2]):
            out[:, j, k] = np.convolve(padded[:, j, k], ker, mode="valid")
    return out


def repair_motion(
    src_pose: np.ndarray,
    src_root: np.ndarray,
    ref_pose: np.ndarray,
    ref_root: np.ndarray,
    upper_blend: float,
    torso_blend: float,
    pose_smooth_win: int,
    root_smooth_win: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    H1 pose_aa body index convention (23):
    0 pelvis(root), 1-10 lower body, 11 torso,
    12-19 arms, 20-21 hands, 22 head.
    """
    n = src_pose.shape[0]
    ref_pose_i = _interp_time(ref_pose, n)
    ref_root_i = _interp_time(ref_root, n)

    pose = src_pose.astype(np.float32, copy=True)
    root = ref_root_i.astype(np.float32, copy=True)

    # Lower body directly uses stable reference to avoid immediate collapse.
    lower_idx = list(range(1, 11))
    for i in lower_idx:
        pose[:, i] = ref_pose_i[:, i]

    # Torso and arms keep source intent but blend toward stable reference.
    pose[:, 11] = torso_blend * src_pose[:, 11] + (1.0 - torso_blend) * ref_pose_i[:, 11]
    for i in [12, 13, 14, 15, 16, 17, 18, 19]:
        pose[:, i] = upper_blend * src_pose[:, i] + (1.0 - upper_blend) * ref_pose_i[:, i]

    # Hands/head have little effect on stability, use stable reference for smoothness.
    for i in [20, 21, 22]:
        pose[:, i] = ref_pose_i[:, i]

    pose = _smooth_3d(pose, pose_smooth_win)
    root = _smooth_2d(root, root_smooth_win)
    return pose, root


def parse_args():
    parser = argparse.ArgumentParser(description="Repair unstable H1 pkl motion by blending with a stable reference.")
    parser.add_argument("--src", required=True, type=str, help="Unstable source pkl.")
    parser.add_argument("--ref", required=True, type=str, help="Stable reference pkl.")
    parser.add_argument("--out", required=True, type=str, help="Output repaired pkl path.")
    parser.add_argument(
        "--upper-blend",
        type=float,
        default=0.50,
        help="Source weight for arm joints [12..19], in [0,1].",
    )
    parser.add_argument(
        "--torso-blend",
        type=float,
        default=0.45,
        help="Source weight for torso joint [11], in [0,1].",
    )
    parser.add_argument("--pose-smooth-win", type=int, default=7, help="Moving-average window for pose_aa.")
    parser.add_argument("--root-smooth-win", type=int, default=7, help="Moving-average window for root_trans_offset.")
    return parser.parse_args()


def main():
    args = parse_args()
    src_path = Path(args.src).resolve()
    ref_path = Path(args.ref).resolve()
    out_path = Path(args.out).resolve()

    src_data, src_key, src_motion = _load_motion(src_path)
    _, _, ref_motion = _load_motion(ref_path)

    src_pose = np.asarray(src_motion["pose_aa"], dtype=np.float32)
    src_root = np.asarray(src_motion["root_trans_offset"], dtype=np.float32)
    ref_pose = np.asarray(ref_motion["pose_aa"], dtype=np.float32)
    ref_root = np.asarray(ref_motion["root_trans_offset"], dtype=np.float32)

    repaired_pose, repaired_root = repair_motion(
        src_pose=src_pose,
        src_root=src_root,
        ref_pose=ref_pose,
        ref_root=ref_root,
        upper_blend=float(args.upper_blend),
        torso_blend=float(args.torso_blend),
        pose_smooth_win=max(1, int(args.pose_smooth_win)),
        root_smooth_win=max(1, int(args.root_smooth_win)),
    )

    out_data = src_data
    out_data[src_key]["pose_aa"] = repaired_pose.astype(np.float32)
    out_data[src_key]["root_trans_offset"] = repaired_root.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out_data, out_path)

    print(f"[DONE] repaired pkl: {out_path}")
    print(
        {
            "src_frames": int(src_pose.shape[0]),
            "ref_frames": int(ref_pose.shape[0]),
            "out_frames": int(repaired_pose.shape[0]),
            "upper_blend": float(args.upper_blend),
            "torso_blend": float(args.torso_blend),
        }
    )


if __name__ == "__main__":
    main()

