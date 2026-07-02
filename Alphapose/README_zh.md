# AlphaPose 交接说明

这个目录只作为 `motion_pipeline` 中的 AlphaPose 说明占位，不建议把完整 AlphaPose 第三方仓库复制进来。

推荐实际目录结构：

```text
F:/LLM-pepper/
  motion_pipeline/
  AlphaPose/
```

## 可以公开提交

```text
README_zh.md
小型路径配置示例
你自己写的非敏感 wrapper 脚本
不含权重、不含视频、不含个人信息的说明文档
```

## 不要公开提交

```text
pretrained_models/
model_files/
detector/yolo/data/*.weights
detector/yolox/data/*.pth
examples/out/
scripts/_623_out/
*.mp4
*.npy
*.pkl
API key
```

## 私下给同门准备

```text
AlphaPose/pretrained_models/       AlphaPose/HybrIK checkpoint
AlphaPose/detector/yolo/data/      yolov3-spp.weights
AlphaPose/detector/yolox/data/     如果使用 YOLOX，放 yolox_l.pth / yolox_x.pth
AlphaPose/model_files/             basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

输出到 `motion_pipeline` 的约定：

```text
motion_pipeline/data/alphapose_raw/<motion_id>/alphapose-results.json
motion_pipeline/data/smpl/smpl_raw/<motion_id>_smpl_raw.npy
```
