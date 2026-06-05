import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def _load_compact_payload(npy_path: Path) -> dict:
    arr = np.load(npy_path, allow_pickle=True)
    if not isinstance(arr, np.ndarray) or arr.dtype != object:
        raise ValueError(f"expected object npy payload, got {type(arr)} dtype={getattr(arr, 'dtype', None)}")
    payload = arr.item()
    if not isinstance(payload, dict) or "frames" not in payload:
        raise ValueError("npy payload must be a dict containing key 'frames'")
    return payload


def _pick_person(frame: dict, person_index: int) -> dict:
    results = frame.get("result", [])
    if not results:
        return {}
    if person_index >= 0 and person_index < len(results):
        return results[person_index]
    return max(results, key=lambda x: float(x.get("bbox_score", 0.0)))


def _collect_smpl_params(payload: dict, person_index: int):
    theta_list = []
    trans_list = []
    shape_list = []
    for i, frame in enumerate(payload["frames"]):
        rec = _pick_person(frame, person_index)
        if not rec:
            raise ValueError(f"frame {i} has no person result")
        theta = np.asarray(rec["pred_theta_mats"], dtype=np.float64)
        transl = np.asarray(rec["transl"], dtype=np.float64)
        shape = np.asarray(rec["pred_shape"], dtype=np.float64)
        if theta.shape != (24, 3, 3):
            raise ValueError(f"frame {i} pred_theta_mats shape {theta.shape}, expected (24, 3, 3)")
        if transl.shape != (3,):
            raise ValueError(f"frame {i} transl shape {transl.shape}, expected (3,)")
        if shape.shape != (10,):
            raise ValueError(f"frame {i} pred_shape shape {shape.shape}, expected (10,)")
        theta_list.append(theta)
        trans_list.append(transl)
        shape_list.append(shape)
    return np.stack(theta_list, axis=0), np.stack(trans_list, axis=0), np.stack(shape_list, axis=0)


def _build_poses(theta_mats: np.ndarray, apply_root_x180: bool) -> np.ndarray:
    t = theta_mats.shape[0]
    rotvec24 = R.from_matrix(theta_mats.reshape(-1, 3, 3)).as_rotvec().reshape(t, 24, 3)
    if apply_root_x180:
        rx180 = R.from_euler("x", 180.0, degrees=True)
        rotvec24[:, 0, :] = (rx180 * R.from_rotvec(rotvec24[:, 0, :])).as_rotvec()
    poses = np.zeros((t, 156), dtype=np.float64)
    poses[:, :72] = rotvec24.reshape(t, 72)
    return poses


def _build_trans(transl: np.ndarray, ref_y0: float) -> np.ndarray:
    trans = transl - transl[0:1]
    trans[:, 1] += ref_y0
    return trans.astype(np.float32)


def _load_reference(ref_npz: Path):
    ref = np.load(ref_npz, allow_pickle=True)
    ref_y0 = float(np.asarray(ref["trans"], dtype=np.float64)[0, 1])
    mocap = ref["mocap_framerate"]
    gender = ref["gender"]
    return ref_y0, mocap, gender


def _print_stats(tag: str, data: dict):
    from scipy.spatial.transform import Rotation as _R

    poses = np.asarray(data["poses"])
    trans = np.asarray(data["trans"])
    root_euler = _R.from_rotvec(poses[:, :3]).as_euler("xyz", degrees=True)
    print(f"[{tag}] frames={poses.shape[0]}")
    print(f"[{tag}] trans.shape={trans.shape} dtype={trans.dtype}, ptp={np.ptp(trans, axis=0)}")
    print(f"[{tag}] poses.shape={poses.shape} dtype={poses.dtype}")
    print(f"[{tag}] root xyz deg mean={root_euler.mean(axis=0)} std={root_euler.std(axis=0)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert AlphaPose compact npy to 0_feats_out-compatible npz.")
    parser.add_argument("--input-npy", required=True, help="Input compact AlphaPose npy path.")
    parser.add_argument("--output-npz", required=True, help="Output npz path.")
    parser.add_argument(
        "--reference-npz",
        default=str(Path(__file__).resolve().parents[1] / "data" / "smplx" / "0_feats_out.npz"),
        help="Reference npz used for y-anchor/gender/mocap_framerate.",
    )
    parser.add_argument("--person-index", type=int, default=-1, help="-1 means choose max bbox_score each frame.")
    parser.add_argument(
        "--no-root-x180-fix",
        action="store_true",
        help="Disable root 180deg rotation around X (camera->upright frame fix).",
    )
    parser.add_argument("--print-ref-stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input_npy).resolve()
    out_path = Path(args.output_npz).resolve()
    ref_path = Path(args.reference_npz).resolve()

    payload = _load_compact_payload(in_path)
    theta_mats, transl, pred_shape = _collect_smpl_params(payload, person_index=args.person_index)
    ref_y0, mocap, gender = _load_reference(ref_path)

    poses = _build_poses(theta_mats, apply_root_x180=not args.no_root_x180_fix)
    trans = _build_trans(transl, ref_y0=ref_y0)
    betas = np.zeros((10,), dtype=np.float64)
    # Keep optional shape signal for debugging parity checks.
    shape_mean = pred_shape.mean(axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        trans=trans.astype(np.float32),
        poses=poses.astype(np.float64),
        gender=gender,
        mocap_framerate=mocap,
        betas=betas,
    )

    out_data = dict(np.load(out_path, allow_pickle=True))
    _print_stats("OUT", out_data)
    print(f"[OUT] betas={out_data['betas']}")
    print(f"[INFO] pred_shape_mean={shape_mean}")
    if args.print_ref_stats:
        _print_stats("REF", dict(np.load(ref_path, allow_pickle=True)))
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
