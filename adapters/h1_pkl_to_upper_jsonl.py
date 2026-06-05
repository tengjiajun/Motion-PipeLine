"""Convert an H1 PHC motion pkl into upper-body jsonl for UpperBodyJsonlCtrl.

Example:
    python scripts/h1_pkl_to_upper_jsonl.py \
        --motion-name singles/3_video2_623 \
        --output-jsonl h1Motion/3_video2_623_upper_body_trajectory.jsonl
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from robojudo.config.h1.ctrl.h1_motion_ctrl_cfg import H1MotionH2HCtrlCfg
from robojudo.config.h1.ctrl.h1_upper_body_jsonl_ctrl_cfg import H1UpperBodyJsonlCtrlCfg
from robojudo.config.h1.env.h1_env_cfg import H1DoF
from robojudo.controller.motion_ctrl import MotionCtrl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert H1 PHC pkl motion to upper-body jsonl trajectory.",
    )
    parser.add_argument(
        "--motion-name",
        type=str,
        required=True,
        help="Motion name under assets/motions/h1/phc, without .pkl (e.g. singles/3_video2_623).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Override output fps. <=0 means use fps stored in pkl (fallback 30).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="pkl_upper_extract",
        help="Value written to each row's source field.",
    )
    return parser.parse_args()


def _load_motion_meta(pkl_path: Path):
    data = joblib.load(pkl_path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Invalid pkl structure: {pkl_path}")
    motion = next(iter(data.values()))
    if "root_trans_offset" not in motion:
        raise ValueError(f"Missing root_trans_offset in {pkl_path}")
    num_frames = int(np.asarray(motion["root_trans_offset"]).shape[0])
    fps = float(motion.get("fps", 30.0))
    return num_frames, fps


def _build_upper_mapping():
    h1_dof_names = H1DoF().joint_names
    jsonl_cfg = H1UpperBodyJsonlCtrlCfg()
    json_joint_names = list(jsonl_cfg.target_joint_names or [])
    if len(json_joint_names) == 0:
        raise ValueError("H1UpperBodyJsonlCtrlCfg.target_joint_names is empty.")

    env_joint_names = [jsonl_cfg.joint_name_map.get(name, name) for name in json_joint_names]
    upper_indices = [h1_dof_names.index(name) for name in env_joint_names]
    return json_joint_names, upper_indices


def main():
    args = parse_args()

    cfg = H1MotionH2HCtrlCfg(motion_name=args.motion_name, motion_ctrl_gui=False)
    pkl_path = Path(cfg.motion_path).resolve()
    if not pkl_path.exists():
        raise FileNotFoundError(f"Motion pkl not found: {pkl_path}")

    num_frames, pkl_fps = _load_motion_meta(pkl_path)
    fps = float(args.fps if args.fps > 0 else pkl_fps)
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    json_joint_names, upper_indices = _build_upper_mapping()

    motion_ctrl = MotionCtrl(cfg_ctrl=cfg, env=None, device="cpu")

    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for frame_idx in range(num_frames):
            t = frame_idx / fps
            motion_ctrl.motion_time = float(t)
            motion_res = motion_ctrl.get_motion()
            dof_pos = motion_res["dof_pos"].detach().cpu().numpy().squeeze().astype(np.float32)
            upper_angles = dof_pos[upper_indices].astype(np.float64).tolist()

            row = {
                "frame_number": int(frame_idx),
                "video_timestamp": float(t),
                "joint_names": json_joint_names,
                "angles": upper_angles,
                "source": args.source,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE] output jsonl: {out_path}")
    print(f"[INFO] pkl: {pkl_path}")
    print(f"[INFO] frames={num_frames}, fps={fps:.6f}, duration={(num_frames / fps):.3f}s")
    print(f"[INFO] joints={json_joint_names}")


if __name__ == "__main__":
    main()

