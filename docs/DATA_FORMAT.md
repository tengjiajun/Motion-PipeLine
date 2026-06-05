# Data Format

## Canonical Motion

Use `canonical_motion.npz` as the only project-owned motion interchange format.

Required fields:

| Field | Shape | Type | Notes |
|---|---:|---|---|
| `fps` | scalar | float32/int | Motion frame rate. |
| `motion_id` | scalar/string | str | Stable id, e.g. `wave_001`. |
| `source_type` | scalar/string | str | `video`, `alphapose`, `smpl_raw`, `manual`, etc. |
| `joint_names` | `[J]` | string array | Canonical robot or body joint names. |
| `q_ref` | `[T, J]` | float32 | Canonical H1 joint reference, radians. |
| `root_pos` | `[T, 3]` | float32 | Optional but preferred, meters. |
| `root_rot` | `[T, 4]` | float32 | Optional quaternion, document order in metadata. |
| `keypoints_3d` | `[T, K, 3]` | float32 | Optional source/body keypoints. |
| `confidence` | `[T, K]` | float32 | Optional keypoint confidence. |

Recommended metadata in sidecar JSON:

```json
{
  "motion_id": "wave_001",
  "fps": 30,
  "quat_order": "xyzw",
  "angle_unit": "rad",
  "coordinate_frame": "z_up",
  "source_video": "data/videos/wave_001.mp4",
  "notes": "Generated from AlphaPose SMPL raw npy."
}
```

## Backend Formats

Backend-specific files are generated artifacts:

- `data/fromw1_inputs/`: FRoM-W1/RoboJuDo input files.
- `data/exbody_inputs/`: ExBody/Isaac Gym input files.

Do not treat backend formats as the source of truth. Regenerate them from canonical motion whenever possible.
