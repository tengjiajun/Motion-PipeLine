import argparse
import sys
from pathlib import Path

import numpy as np
import torch


SMPL_PARENT = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]


def _pick_person(frame: dict, person_index: int):
    people = frame.get("result", [])
    if not people:
        return None
    if 0 <= person_index < len(people):
        return people[person_index]
    return max(people, key=lambda item: float(item.get("bbox_score", 0.0)))


def _interp_nan_cols(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    t = np.arange(flat.shape[0], dtype=np.float32)
    for c in range(flat.shape[1]):
        v = flat[:, c]
        m = np.isfinite(v)
        if not np.any(m):
            flat[:, c] = 0.0
            continue
        if np.count_nonzero(m) == 1:
            flat[:, c] = v[m][0]
            continue
        flat[:, c] = np.interp(t, t[m], v[m]).astype(np.float32)
    return flat.reshape(arr.shape)


def _load_body_and_theta(smpl_npy: Path, person_index: int, use_transl: bool):
    payload = np.load(str(smpl_npy), allow_pickle=True).item()
    frames = payload["frames"]
    t = len(frames)
    body24 = np.full((t, 24, 3), np.nan, dtype=np.float32)
    theta = np.full((t, 24, 3, 3), np.nan, dtype=np.float32)

    for i, frame in enumerate(frames):
        person = _pick_person(frame, person_index)
        if person is None:
            continue
        j24 = np.asarray(person["pred_xyz_jts_24"], dtype=np.float32)
        if use_transl:
            j24 = j24 + np.asarray(person["transl"], dtype=np.float32)[None, :]
        body24[i] = j24
        theta[i] = np.asarray(person["pred_theta_mats"], dtype=np.float32)

    body24 = _interp_nan_cols(body24)
    theta = _interp_nan_cols(theta)
    return body24, theta


def _build_hand_template(is_right: bool) -> np.ndarray:
    sign = -1.0 if is_right else 1.0
    finger_dirs = {
        "index": np.array([sign, 0.12, 0.05], dtype=np.float32),
        "middle": np.array([sign, 0.00, 0.07], dtype=np.float32),
        "pinky": np.array([sign, -0.18, 0.02], dtype=np.float32),
        "ring": np.array([sign, -0.10, 0.05], dtype=np.float32),
        "thumb": np.array([sign * 0.70, 0.22, -0.10], dtype=np.float32),
    }
    segment_lengths = [0.026, 0.020, 0.016]
    roots = {}
    for name, vec in finger_dirs.items():
        direction = vec / (np.linalg.norm(vec) + 1e-8)
        roots[name] = direction * 0.035

    order = ["index", "middle", "pinky", "ring", "thumb"]
    out = []
    for name in order:
        direction = finger_dirs[name] / (np.linalg.norm(finger_dirs[name]) + 1e-8)
        point = roots[name].copy()
        for length in segment_lengths:
            point = point + direction * length
            out.append(point.copy())
    return np.asarray(out, dtype=np.float32)


def _local_to_global_rot(local24: np.ndarray) -> np.ndarray:
    out = np.zeros_like(local24, dtype=np.float32)
    for i, parent in enumerate(SMPL_PARENT):
        if parent < 0:
            out[i] = local24[i]
        else:
            out[i] = out[parent] @ local24[i]
    return out


def _make_joints52(body24: np.ndarray, theta: np.ndarray) -> np.ndarray:
    t = body24.shape[0]
    joints = np.zeros((t, 52, 3), dtype=np.float32)
    joints[:, :22] = body24[:, :22]

    hand_l = _build_hand_template(is_right=False)
    hand_r = _build_hand_template(is_right=True)

    for i in range(t):
        grot = _local_to_global_rot(theta[i])
        l_wrist = joints[i, 20]
        r_wrist = joints[i, 21]
        joints[i, 22:37] = l_wrist + (grot[20] @ hand_l.T).T
        joints[i, 37:52] = r_wrist + (grot[21] @ hand_r.T).T
    return joints


def _setup_motionx(fromw1_root: Path):
    h_gpt_root = fromw1_root / "H-GPT"
    if str(h_gpt_root) not in sys.path:
        sys.path.insert(0, str(h_gpt_root))
    from hGPT.data.motionx.scripts import motion_process as mp  # noqa: WPS433

    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.n_raw_offsets = torch.from_numpy(mp.t2m_raw_offsets)
    mp.kinematic_chain = mp.t2m_body_hand_kinematic_chain
    return mp


def _load_norm_meta(fromw1_root: Path):
    meta_dir = fromw1_root / "H-ACT" / "retarget" / "assets" / "meta"
    mean = np.load(str(meta_dir / "mean.npy")).astype(np.float32)
    std = np.load(str(meta_dir / "std.npy")).astype(np.float32)
    return mean, std


def _set_target_offsets(mp, joints52: np.ndarray, reference_623: Path, fromw1_root: Path):
    tgt_skel = mp.Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
    if reference_623.exists():
        ref = np.load(str(reference_623))
        if ref.ndim != 3 or ref.shape[0] != 1 or ref.shape[-1] != 623:
            raise ValueError(f"reference 623 must be (1, T, 623), got {ref.shape}")
        mean, std = _load_norm_meta(fromw1_root)
        ref_denorm = ref[0].astype(np.float32) * std + mean
        ref_j = mp.recover_from_ric(torch.from_numpy(ref_denorm).unsqueeze(0).float(), 52)
        mp.tgt_offsets = tgt_skel.get_offsets_joints(ref_j[0, 0])
    else:
        mp.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(joints52[0]))


def _joints52_to_623(
    joints52: np.ndarray,
    fromw1_root: Path,
    feet_thre: float,
    reference_623: Path,
) -> np.ndarray:
    mp = _setup_motionx(fromw1_root=fromw1_root)
    _set_target_offsets(mp=mp, joints52=joints52, reference_623=reference_623, fromw1_root=fromw1_root)
    data, _, _, _ = mp.process_file(joints52.astype(np.float32), feet_thre)

    mean, std = _load_norm_meta(fromw1_root)
    norm = (data.astype(np.float32) - mean) / np.maximum(std, 1e-8)
    return np.expand_dims(norm.astype(np.float32), 0)


def _recover_ref_joints52(reference_623: Path, fromw1_root: Path) -> np.ndarray:
    if not reference_623.exists():
        raise FileNotFoundError(f"reference 623 not found: {reference_623}")
    ref = np.load(str(reference_623))
    if ref.ndim != 3 or ref.shape[0] != 1 or ref.shape[-1] != 623:
        raise ValueError(f"reference 623 must be (1, T, 623), got {ref.shape}")

    h_gpt_root = fromw1_root / "H-GPT"
    if str(h_gpt_root) not in sys.path:
        sys.path.insert(0, str(h_gpt_root))
    from hGPT.data.motionx.scripts.motion_process import recover_from_ric  # noqa: WPS433

    mean, std = _load_norm_meta(fromw1_root)
    ref_denorm = ref[0].astype(np.float32) * std + mean
    joints = recover_from_ric(torch.from_numpy(ref_denorm).unsqueeze(0).float(), 52)
    return joints.squeeze(0).cpu().numpy().astype(np.float32)


def _yaw_from_lr(vec_l_to_r_xz: np.ndarray) -> float:
    x, z = float(vec_l_to_r_xz[0]), float(vec_l_to_r_xz[1])
    if abs(x) + abs(z) < 1e-8:
        return 0.0
    return float(np.arctan2(z, x))


def _rotz_y(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.asarray(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _stabilize_lower_body(joints52: np.ndarray, reference_623: Path, fromw1_root: Path) -> np.ndarray:
    out = joints52.copy()
    ref = _recover_ref_joints52(reference_623=reference_623, fromw1_root=fromw1_root)
    ref0 = ref[0]

    lower_idx = [1, 2, 4, 5, 7, 8, 10, 11]
    pelvis_ref = ref0[0].copy()
    ref_rel = ref0[lower_idx] - pelvis_ref[None, :]
    lr_ref = ref0[1] - ref0[2]
    yaw_ref = _yaw_from_lr(np.asarray([lr_ref[0], lr_ref[2]], dtype=np.float32))

    for t in range(out.shape[0]):
        pelvis = out[t, 0].copy()
        lr = out[t, 1] - out[t, 2]
        yaw_src = _yaw_from_lr(np.asarray([lr[0], lr[2]], dtype=np.float32))
        rot = _rotz_y(yaw_src - yaw_ref)
        out[t, lower_idx] = (ref_rel @ rot.T) + pelvis[None, :]

    return out


def smpl_npy_to_623(
    input_npy: Path,
    output_npy: Path,
    fromw1_root: Path,
    reference_623: Path,
    person_index: int = -1,
    use_transl: bool = False,
    cv_y_flip: bool = True,
    cv_z_flip: bool = True,
    stabilize_lower_body: bool = True,
    feet_thre: float = 0.002,
) -> np.ndarray:
    body24, theta = _load_body_and_theta(
        smpl_npy=input_npy,
        person_index=person_index,
        use_transl=use_transl,
    )
    joints52 = _make_joints52(body24=body24, theta=theta)
    if cv_y_flip:
        joints52[..., 1] *= -1.0
    if cv_z_flip:
        joints52[..., 2] *= -1.0
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


def _clean_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_smpl_repaired_compact", "_smpl_repaired", "_smpl_raw"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _iter_inputs(args) -> list[Path]:
    if args.input_npy:
        return [args.input_npy.resolve()]
    input_dir = args.input_dir.resolve()
    return sorted(input_dir.glob(args.glob))


def _resolve_reference_623(args) -> Path:
    if args.reference_623 is not None:
        return args.reference_623.resolve()
    return (args.fromw1_root / "H-ACT" / "retarget" / "data" / "623" / "0_feats_out.npy").resolve()


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "data" / "fromw1_inputs"

    parser = argparse.ArgumentParser(
        description="Convert AlphaPose SMPL npy to FRoM-W1 normalized 623 feature npy."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npy", type=Path, help="Input AlphaPose SMPL npy.")
    source.add_argument("--input-dir", type=Path, help="Batch input directory.")
    parser.add_argument("--glob", default="*_smpl_raw.npy", help="Batch glob used with --input-dir.")
    parser.add_argument("--output-npy", type=Path, default=None, help="Single output .npy path.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Batch/default output directory.")
    parser.add_argument("--person-index", type=int, default=-1, help="-1 means choose max bbox_score each frame.")
    parser.add_argument(
        "--use-transl",
        action="store_true",
        help="Add AlphaPose transl to body joints. Default is off to avoid camera-depth drift.",
    )
    parser.add_argument(
        "--no-use-transl",
        action="store_true",
        help="Deprecated alias; translation is disabled by default.",
    )
    parser.add_argument(
        "--no-cv-y-flip",
        action="store_true",
        help="Disable camera-to-motion Y-axis flip.",
    )
    parser.add_argument(
        "--no-cv-z-flip",
        action="store_true",
        help="Disable camera-depth Z-axis flip. Default fixes AlphaPose camera forward vs MotionX forward.",
    )
    parser.add_argument(
        "--no-stabilize-lower-body",
        action="store_true",
        help="Disable lower-body stabilization with reference standing template.",
    )
    parser.add_argument("--feet-thre", type=float, default=0.002, help="Foot contact threshold.")
    parser.add_argument(
        "--fromw1-root",
        type=Path,
        default=repo_root.parent / "FRoM-W1",
        help="FRoM-W1 root; used to import H-GPT motion_process.",
    )
    parser.add_argument(
        "--reference-623",
        type=Path,
        default=None,
        help="Reference normalized 623 file for target offsets/lower-body template.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.fromw1_root = args.fromw1_root.resolve()
    reference_623 = _resolve_reference_623(args)
    input_paths = _iter_inputs(args)
    if not input_paths:
        raise FileNotFoundError("No input npy files matched.")
    if len(input_paths) > 1 and args.output_npy:
        raise ValueError("--output-npy can only be used with one input.")

    for input_npy in input_paths:
        if args.output_npy is not None:
            output_npy = args.output_npy.resolve()
        else:
            output_npy = (
                args.output_dir / f"{_clean_stem(input_npy)}_623.npy"
            ).resolve()
        features_623 = smpl_npy_to_623(
            input_npy=input_npy,
            output_npy=output_npy,
            fromw1_root=args.fromw1_root,
            reference_623=reference_623,
            person_index=args.person_index,
            use_transl=bool(args.use_transl and not args.no_use_transl),
            cv_y_flip=not args.no_cv_y_flip,
            cv_z_flip=not args.no_cv_z_flip,
            stabilize_lower_body=not args.no_stabilize_lower_body,
            feet_thre=args.feet_thre,
        )
        print(f"[DONE] {input_npy} -> {output_npy} shape={features_623.shape} dtype={features_623.dtype}")


if __name__ == "__main__":
    main()
