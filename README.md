# motion_pipeline

This repository is the orchestration layer for the motion workflow. AlphaPose, FRoM-W1, and ExBody stay in their own projects and environments. This project owns shared data contracts, manifests, adapters, logs, and evaluation outputs.

## Role

```text
video
  -> AlphaPose / SMPL raw npy
  -> human canonical motion
  -> FRoM-W1 input
  -> ExBody input
  -> h1_reference_motion.npz
  -> backend results + metrics
```

## Layout

```text
adapters/        Format conversion wrappers.
configs/         Machine-specific and pipeline configs.
docs/            Data format, pipeline, sync, and experiment notes.
manifests/       Per-motion records and batch manifests.
scripts/         Entry points for batch conversion/evaluation.
tools/           Small shared utilities.
data/            Local generated data. Large contents are gitignored.
logs/            Runtime logs. Gitignored.
```

## Motion Layers

`canonical_motion.npz` is the cleaned human-motion layer. The robot-level reference layer is
`h1_reference_motion.npz`, which stores H1 root pose and 19-DoF joint references before any
RoboJuDo/Isaac/MuJoCo execution. LLM edits should target the H1 reference layer when the goal is
to refine robot behavior rather than repair AlphaPose/SMPL estimates.

## First rule

Do not edit AlphaPose, FRoM-W1, or ExBody internals just to connect the pipeline. Add wrappers/adapters here and exchange files through canonical motion and manifests.
