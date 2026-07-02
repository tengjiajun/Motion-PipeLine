# FRoM-W1 交接说明

这个目录只作为 `motion_pipeline` 中的 FRoM-W1 说明占位，不建议把完整 FRoM-W1 第三方仓库复制进来。

推荐实际目录结构：

```text
F:/LLM-pepper/
  motion_pipeline/
  FRoM-W1/
```

本项目目前主要调用：

```text
FRoM-W1/H-ACT/retarget/
```

用于把 623D 人体动作特征 retarget 成 H1 pkl。

## 可以公开提交

```text
README_zh.md
小型路径配置示例
你自己写的非敏感 wrapper 脚本
不含权重、不含 SMPL/MANO、不含生成数据的说明文档
```

## 不要公开提交

```text
H-ACT/retarget/models/
H-ACT/retarget/assets/
H-GPT/experiments/
datasets/
*.pth
*.pt
*.ckpt
*.bin
*.pkl
*.npy
*.npz
```

## 私下给同门准备

```text
FRoM-W1/H-ACT/retarget/models/smpl/
  SMPL_NEUTRAL.pkl
  SMPL_MALE.pkl
  SMPL_FEMALE.pkl

FRoM-W1/H-ACT/retarget/models/mano/
  MANO_LEFT.pkl
  MANO_RIGHT.pkl

FRoM-W1/H-ACT/retarget/assets/
  beta/
  meta/
  robot/
    h1/
    g1/
    dex3/
    inspire/
```

`motion_pipeline` 调用 FRoM-W1 的主要脚本：

```text
adapters/canonical_to_fromw1_623.py
tools/fromw1_623_to_pkl_batch.py
adapters/fromw1_pkl_to_h1_reference.py
tools/h1_reference_to_gif.py
```
