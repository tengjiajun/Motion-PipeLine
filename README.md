# motion_pipeline

This repository is the orchestration layer for the motion workflow. AlphaPose, FRoM-W1, and ExBody stay in their own projects and environments. This project owns shared data contracts, manifests, adapters, logs, and evaluation outputs.

## Role

```text
video
  -> AlphaPose / SMPL raw npy
  -> canonical_motion.npz
  -> FRoM-W1 input
  -> ExBody input
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

## First rule

Do not edit AlphaPose, FRoM-W1, or ExBody internals just to connect the pipeline. Add wrappers/adapters here and exchange files through canonical motion and manifests.
