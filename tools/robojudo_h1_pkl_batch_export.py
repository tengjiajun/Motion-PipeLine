import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import mujoco
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch run H1 pkl motions in RoboJuDo and export rollout data plus optional GIFs."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing H1 pkl files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write outputs.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--gif-fps", type=int, default=25)
    parser.add_argument("--capture-every-n-steps", type=int, default=2)
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF rendering and only export data.")
    return parser.parse_args()


def prepare_runtime_home(robojudo_root: Path):
    runtime_home = (robojudo_root / ".runtime_home").resolve()
    (runtime_home / ".config").mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = runtime_home.as_posix()
    os.environ["USERPROFILE"] = runtime_home.as_posix()
    os.environ["XDG_CONFIG_HOME"] = (runtime_home / ".config").as_posix()


def save_gif(frames, out_path: Path, fps: int):
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_rollout_csv(run_data: dict, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    joint_names = run_data["joint_names"]
    header = ["step", "time_s"]
    for name in joint_names:
        header.append(f"pd_target__{name}")
    for name in joint_names:
        header.append(f"dof_pos__{name}")
    header.extend(["base_pos_x", "base_pos_y", "base_pos_z"])
    header.extend(["base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"])
    for name in joint_names:
        header.append(f"ref_dof_pos__{name}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(run_data["time_s"])):
            row = [i, float(run_data["time_s"][i])]
            row.extend(run_data["pd_target"][i].tolist())
            row.extend(run_data["dof_pos"][i].tolist())
            row.extend(run_data["base_pos"][i].tolist())
            row.extend(run_data["base_quat"][i].tolist())
            row.extend(run_data["ref_dof_pos"][i].tolist())
            writer.writerow(row)


def save_rollout_npz(run_data: dict, out_npz: Path):
    np.savez_compressed(
        out_npz,
        time_s=run_data["time_s"],
        pd_target=run_data["pd_target"],
        dof_pos=run_data["dof_pos"],
        base_pos=run_data["base_pos"],
        base_quat=run_data["base_quat"],
        ref_dof_pos=run_data["ref_dof_pos"],
        joint_names=np.asarray(run_data["joint_names"], dtype=object),
        dt=np.asarray([run_data["dt"]], dtype=np.float32),
    )


def main():
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    robojudo_root = repo_root.parent / "RoboJuDo"
    prepare_runtime_home(robojudo_root=robojudo_root)

    import sys

    if str(robojudo_root) not in sys.path:
        sys.path.insert(0, str(robojudo_root))

    from robojudo.config.h1.h1_cfg import h1_h2h
    from robojudo.config.h1.ctrl.h1_motion_ctrl_cfg import H1MotionH2HCtrlCfg
    from robojudo.pipeline.rl_pipeline import RlPipeline

    class AbsoluteH1MotionH2HCtrlCfg(H1MotionH2HCtrlCfg):
        motion_file: str = ""

        @property
        def motion_path(self) -> str:
            return Path(self.motion_file).resolve().as_posix()

    def close_viewer_if_any(pipeline: RlPipeline):
        viewer = getattr(pipeline.env, "viewer", None)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    def build_pipeline(pkl_path: Path):
        cfg = h1_h2h()
        cfg.ctrl = [AbsoluteH1MotionH2HCtrlCfg(motion_file=str(pkl_path), motion_ctrl_gui=False)]
        cfg.run_fullspeed = True
        cfg.do_safety_check = False
        return RlPipeline(cfg)

    def enable_motion_playback(pipeline: RlPipeline):
        ctrl_bundle = getattr(pipeline.ctrl_manager, "controllers", None)
        if ctrl_bundle is None:
            return
        motion_ctrl_cfg = getattr(ctrl_bundle, "MotionH2HCtrl", None)
        if motion_ctrl_cfg is None:
            return
        motion_ctrl = motion_ctrl_cfg.inst
        motion_ctrl.speed_target = 1.0
        motion_ctrl.play_speed_ratio = 1.0

    def run_rollout(pipeline: RlPipeline, duration_s: float):
        close_viewer_if_any(pipeline)
        env_joint_names = list(pipeline.env.joint_names)
        dt = float(pipeline.dt)
        max_steps = max(1, int(np.floor(duration_s / dt)))

        pd_target_all = []
        dof_pos_all = []
        base_pos_all = []
        base_quat_all = []
        ref_dof_pos_all = []

        for _ in range(max_steps):
            pipeline.env.update()
            env_data = pipeline.env.get_data()
            ctrl_data = pipeline.ctrl_manager.get_ctrl_data(env_data)

            obs, extras = pipeline.policy.get_observation(env_data, ctrl_data)
            pd_target = pipeline.policy.get_pd_target(obs)
            pipeline.env.step(pd_target, extras.get("hand_pose", None))
            pipeline.post_step_callback(env_data, ctrl_data, extras, pd_target)

            ref_dof = np.full((len(env_joint_names),), np.nan, dtype=np.float32)
            if "MotionH2HCtrl" in ctrl_data:
                ref_dof = np.asarray(ctrl_data["MotionH2HCtrl"]["dof_pos"], dtype=np.float32)

            pd_target_all.append(pd_target.copy())
            dof_pos_all.append(pipeline.env.dof_pos.copy())
            base_pos_all.append(np.asarray(pipeline.env.base_pos, dtype=np.float32).copy())
            base_quat_all.append(np.asarray(pipeline.env.base_quat, dtype=np.float32).copy())
            ref_dof_pos_all.append(ref_dof.copy())

        return {
            "joint_names": env_joint_names,
            "time_s": np.arange(max_steps, dtype=np.float32) * dt,
            "pd_target": np.asarray(pd_target_all, dtype=np.float32),
            "dof_pos": np.asarray(dof_pos_all, dtype=np.float32),
            "base_pos": np.asarray(base_pos_all, dtype=np.float32),
            "base_quat": np.asarray(base_quat_all, dtype=np.float32),
            "ref_dof_pos": np.asarray(ref_dof_pos_all, dtype=np.float32),
            "dt": dt,
        }

    def render_rollout_to_frames(pipeline: RlPipeline, duration_s: float):
        close_viewer_if_any(pipeline)
        renderer = mujoco.Renderer(pipeline.env.model, height=args.height, width=args.width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth = 180.0
        cam.elevation = -15.0
        cam.distance = 2.8

        dt = pipeline.dt
        total_steps = max(1, int(np.floor(duration_s / dt)))
        capture_every = max(1, int(args.capture_every_n_steps))
        frames = []
        for step in range(total_steps):
            pipeline.step()
            if step % capture_every != 0:
                continue
            if pipeline.env.base_pos is not None:
                cam.lookat[:] = pipeline.env.base_pos
            renderer.update_scene(pipeline.env.data, cam)
            frames.append(renderer.render())
        renderer.close()
        return frames

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pkl_path in sorted(args.input_dir.glob("*.pkl")):
        motion = joblib.load(pkl_path)
        motion0 = next(iter(motion.values())) if isinstance(motion, dict) else motion
        duration_s = float(motion0["root_trans_offset"].shape[0]) / float(motion0.get("fps", 30.0))
        stem = pkl_path.stem
        motion_dir = args.output_dir / stem
        motion_dir.mkdir(parents=True, exist_ok=True)

        data_pipeline = None
        gif_pipeline = None
        try:
            print(f"[RUN] {pkl_path.name}", flush=True)
            data_pipeline = build_pipeline(pkl_path)
            enable_motion_playback(data_pipeline)
            run_data = run_rollout(data_pipeline, duration_s=duration_s)
            save_rollout_csv(run_data, motion_dir / "rollout.csv")
            save_rollout_npz(run_data, motion_dir / "rollout.npz")
            with open(motion_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input_pkl": str(pkl_path),
                        "duration_s": duration_s,
                        "dt": run_data["dt"],
                        "steps": int(len(run_data["time_s"])),
                        "joint_names": run_data["joint_names"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[DATA] {motion_dir}", flush=True)

            if not args.no_gif:
                gif_pipeline = build_pipeline(pkl_path)
                enable_motion_playback(gif_pipeline)
                frames = render_rollout_to_frames(gif_pipeline, duration_s=duration_s)
                save_gif(frames, args.output_dir / f"{stem}_h1_h2h.gif", fps=args.gif_fps)
                print(f"[GIF] {stem}_h1_h2h.gif frames={len(frames)}", flush=True)
        finally:
            for pipeline in (data_pipeline, gif_pipeline):
                if pipeline is not None:
                    try:
                        pipeline.env.shutdown()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
