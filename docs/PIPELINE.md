# Pipeline

## Stages

| Stage | Input | Output | Usual machine |
|---|---|---|---|
| `alphapose` | `data/videos/*.mp4` | `data/alphapose_raw/`, `data/smpl_raw/` | Windows |
| `canonicalize` | `data/smpl_raw/*.npy` | `data/canonical/*.npz` | Windows or Ubuntu |
| `export_fromw1` | `data/canonical/*.npz` | `data/fromw1_inputs/*` | Windows |
| `export_exbody` | `data/canonical/*.npz` | `data/exbody_inputs/*` | Ubuntu |
| `run_fromw1` | `data/fromw1_inputs/*` | `data/results/fromw1/*` | Windows |
| `run_exbody` | `data/exbody_inputs/*` | `data/results/exbody/*` | Ubuntu |
| `metrics` | backend result videos/logs | `data/metrics/*.json` | Either |

## Manifest State

Each motion should have one manifest record. Update it after each stage instead of relying on chat history.

Minimum states:

```text
pending -> running -> done -> failed
```

Failure records should include the command, machine, environment, and short error summary.
