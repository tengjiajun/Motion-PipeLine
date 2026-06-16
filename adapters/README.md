# Adapters

Format conversion and cleanup scripts used by the orchestration layer.

## Current Files

| File | Source | Purpose | Notes |
|---|---|---|---|
| `repair_smpl_npy.py` | AlphaPose | Repair/interpolate AlphaPose SMPL object npy. | Pure numpy/scipy. |
| `drop_missing_smpl_frames.py` | AlphaPose | Drop frames with no detected person. | Pure numpy. |
| `alphapose_npy_to_fromw1_ref_npz.py` | FRoM-W1 | Convert AlphaPose compact SMPL npy to FRoM-W1 retarget-compatible SMPL npz. | Pass `--reference-npz` explicitly from FRoM-W1 assets. |
| `smpl_npy_to_fromw1_623.py` | FRoM-W1 | Convert AlphaPose SMPL npy to normalized FRoM-W1 623 feature npy. | Applies AlphaPose camera Y/Z flips by default; requires `--fromw1-root` when FRoM-W1 is not beside this repo. |
| `canonical_to_fromw1_623.py` | motion_pipeline | Convert canonical MotionX-52 joints to FRoM-W1 normalized 623 feature npy. | Preferred path after `canonical_motion.npz` exists. |
| `canonical_to_exbody_h1.py` | motion_pipeline | Convert canonical SMPL rotations to ExBody/Isaac Gym H1 retarget files. | Run in the ExBody environment with `poselib`; pass `--exbody-root`. |
| `fromw1_pkl_to_h1_reference.py` | FRoM-W1 | Convert retargeted H1 pkl to backend-independent `h1_reference_motion.npz`. | Preferred robot-level reference format for LLM edits. |
| `h1_reference_to_fromw1_pkl.py` | FRoM-W1/RoboJuDo | Convert edited `h1_reference_motion.npz` back to FRoM-W1 pkl. | Keeps existing GIF/RoboJuDo execution tools usable. |
| `fromw1_pkl_to_beyondmimic_csv.py` | FRoM-W1 | Convert retargeted pkl motion to BeyondMimic CSV. | Batch defaults may need path edits later. |
| `repair_h1_pkl_stability.py` | RoboJuDo | Blend unstable H1 pkl with stable reference. | Pure joblib/numpy. |
| `h1_pkl_to_upper_jsonl.py` | RoboJuDo | Extract H1 upper-body joint trajectory JSONL from PHC pkl. | Requires RoboJuDo import path/env. |
| `limit_h1_jsonl_velocity.py` | RoboJuDo | Clamp H1 upper-body JSONL limits and velocity. | Pure Python. |

## Deliberately Not Copied

- RoboJuDo `video_mp4_to_623_one_shot.py`: overlaps with pipeline stage orchestration.
- RoboJuDo `alphapose_npy_to_fromw1_623.py`: replaced by `smpl_npy_to_fromw1_623.py` for direct SMPL-to-623 export.
- FRoM-W1 `npz_2_stickman_gif.py`: depends on FRoM-W1 retarget package layout; keep it in FRoM-W1 until wrapped cleanly.
- RoboJuDo run/render scripts: they execute backend simulations and should stay in RoboJuDo, called by wrappers later.
