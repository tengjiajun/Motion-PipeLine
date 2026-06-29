# H1 Motion Review UI

Run from the `motion_pipeline` repository:

```powershell
python frontend/server.py --open
```

The default address is `http://127.0.0.1:8765`.

The server scans `data/` for source videos, H1 reference GIFs, execution GIFs,
Qwen-VL qualitative reports, and Qwen-Max edit plans. Submitted user feedback is
stored under:

```text
data/feedback/<backend>/<motion_id>/<version>/
```

Use a different data directory when running on another operating system:

```powershell
python frontend/server.py --data-dir D:\shared\motion_pipeline\data
```

```bash
python frontend/server.py --data-dir /mnt/shared/motion_pipeline/data
```
