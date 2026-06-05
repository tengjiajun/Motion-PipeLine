# Scripts

Entry-point wrappers for batch processing. Scripts should read manifests and machine-local configs instead of hardcoded absolute paths.

## Current Entry Points

| File | Purpose |
|---|---|
| `evaluate_alphapose_smpl.py` | Generate AlphaPose/SMPL source-layer quality reports and LLM text. |
| `repair_alphapose_smpl_batch.py` | Batch repair `smpl_raw` into `smpl_repaired` and `smpl_repaired_compact`. |
| `smpl_repaired_to_canonical.py` | Convert repaired compact SMPL files into `canonical_motion.npz`. |
| `evaluate_canonical_motion.py` | Evaluate canonical/reference smoothness, stability, semantic quality, and LLM edit hints. |
| `repair_canonical_motion.py` | Apply generic canonical reference smoothing and lower-body stabilization. |
