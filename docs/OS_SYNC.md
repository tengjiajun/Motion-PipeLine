# Sharing Files Across Windows and Ubuntu

## Recommended Setup

Use a dedicated shared data location that both systems can read/write, plus Git for code/docs.

Best options:

| Option | Use for | Notes |
|---|---|---|
| Separate NTFS/exFAT data partition | Large files and generated artifacts | Good for dual boot. Disable Windows Fast Startup if using NTFS from Ubuntu. |
| External SSD, exFAT | Large files moved between systems | Simple and robust. |
| Git | Code, docs, small configs, manifests | Do not commit videos/checkpoints/large npy files. |
| Git LFS or DVC | Versioned large datasets | Useful later for paper reproducibility. |
| Cloud drive/NAS | Backups and remote transfer | Be careful with partial sync while files are being written. |

## Practical Recommendation

Keep the same relative structure on both OSes:

```text
motion_pipeline/
  data/canonical/wave_001.npz
  manifests/wave_001.json
```

Avoid hardcoding absolute paths like:

```text
F:\LLM-pepper\...
/home/user/...
```

Use machine config files instead:

```text
configs/windows.local.json
configs/ubuntu.local.json
```

Example:

```json
{
  "repo_root": "F:/LLM-pepper/motion_pipeline",
  "alphapose_root": "F:/LLM-pepper/AlphaPose",
  "fromw1_root": "F:/LLM-pepper/FRoM-W1",
  "exbody_root": null
}
```

```json
{
  "repo_root": "/home/user/motion_pipeline",
  "alphapose_root": null,
  "fromw1_root": null,
  "exbody_root": "/home/user/expressive-humanoid"
}
```

## Dual Boot NTFS Warning

If Ubuntu writes to an NTFS Windows partition, disable Windows Fast Startup and fully shut down Windows before booting Ubuntu. Otherwise Ubuntu may mount the partition read-only or risk filesystem inconsistency.

## Handoff Habit

Before rebooting to the other OS, update:

- `docs/HANDOFF.md`
- related `manifests/*.json`
- `docs/EXPERIMENT_LOG.md`
