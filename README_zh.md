# motion_pipeline 中文说明

本项目是“视频动作到 H1 机器人动作模仿与迭代评估”的集成层。它不直接替代 AlphaPose、FRoM-W1 或 RoboJuDo，而是负责把这些项目的输入输出串起来，统一管理数据格式、转换脚本、评估结果、可视化界面和大模型反馈。

当前共享给同门时，建议先只复现这条链路：

```text
原始视频
  -> AlphaPose / HybrIK 生成 SMPL raw npy
  -> 修复和压缩 SMPL
  -> 生成 canonical motion
  -> 转换为 FRoM-W1 623D 输入
  -> FRoM-W1 retarget 生成 H1 pkl
  -> 转换为 h1_reference_motion.npz
  -> 生成 H1 参考动作 GIF
  -> RoboJuDo 执行并录制实际动作 GIF
  -> 前端对比 + 用户评价 + 千问视觉/文本模型评价
```

ExBody/Isaac Gym 路线暂时先不作为同门复现目标。

## 1. 推荐目录结构

建议把几个项目放在同一个父目录下，例如：

```text
F:/LLM-pepper/
  motion_pipeline/      # 本仓库，负责集成、转换、评估、前端
  AlphaPose/            # 第三方 AlphaPose 仓库，单独安装
  FRoM-W1/              # 第三方 FRoM-W1 仓库，单独安装
  RoboJuDo/             # 用于执行 H1 pkl，可后续再配置
```

本仓库中的 `Alphapose/` 和 `FRoM-W1/` 目录只是占位说明目录，不建议直接把完整第三方仓库复制进来。第三方项目最好作为并列目录、Git submodule，或者让同门自己按 README 下载。

## 2. 本仓库结构

```text
adapters/        格式转换和数据修复脚本
configs/         本机路径配置示例
docs/            数据格式、跨系统同步、实验记录说明
frontend/        本地动作对比和反馈前端
manifests/       单个动作的元数据记录
scripts/         批量评估、修复、大模型调用等脚本
tools/           GIF 渲染、FRoM-W1 retarget 批处理等工具
data/            本地生成数据，默认大部分不应上传 GitHub
logs/            运行日志，默认不上传
Alphapose/       AlphaPose 交接说明，不放权重和大数据
FRoM-W1/         FRoM-W1 交接说明，不放权重和大数据
```

核心数据层级：

```text
data/videos/                       原始视频
data/alphapose_raw/                AlphaPose JSON 输出
data/smpl/smpl_raw/                AlphaPose/HybrIK 导出的 SMPL raw npy
data/smpl/smpl_repaired/           修复后的 SMPL
data/smpl/smpl_repaired_compact/   删除无人体帧后的 SMPL
data/canonical/                    人体动作中间层
data/fromw1_inputs/                FRoM-W1 623D 输入
data/fromw1_pkl/                   FRoM-W1 retarget 后的 H1 pkl
data/h1_reference/                 机器人层 H1 参考动作
data/gifs/                         原视频/参考动作/执行动作的可视化
data/results/robojudo/             RoboJuDo 执行结果
data/metrics/                      AlphaPose、canonical、执行层评估
data/llm/                          千问视觉模型和文本模型输出
data/feedback/                     前端保存的用户反馈
```

## 3. GitHub 可以直接上传什么

可以上传：

```text
adapters/
configs/*.example.json
docs/
frontend/
manifests/example_motion.json
scripts/
tools/
Alphapose/README_zh.md
FRoM-W1/README_zh.md
README.md
README_zh.md
run_motion_review_app.bat
.gitignore
LICENSE（如果你后续添加自己的许可证）
```

可以选择性上传的小型示例：

```text
data/metrics/ 下的少量 json/csv/md 评估样例
data/canonical/ 下 1 个不含隐私的 demo npz
data/h1_reference/ 下 1 个不含隐私的 demo npz
data/gifs/ 下 1 组 demo gif
```

不建议公开上传：

```text
*.mp4, *.avi, *.mov                原始人物视频，可能涉及隐私
*.npy, *.npz, *.pkl                大量生成数据和动作数据
*.pth, *.pt, *.ckpt, *.bin         模型权重
SMPL/MANO/SMPL-X 模型文件          需要遵守官方许可
FRoM-W1 retarget assets            需要遵守原项目/数据集许可
qw_LLM.txt、API Key、.env          密钥绝不能上传
logs/                              本机路径和临时日志
```

当前 `.gitignore` 已经默认忽略大部分图片、视频、npy、npz、pkl、日志和环境文件。发布前建议运行：

```powershell
git status --short
```

确认没有误提交权重、视频、密钥和本机绝对路径。

## 4. 需要私下发给同门的文件

建议整理成一个压缩包，例如 `motion_pipeline_private_assets_YYYYMMDD.zip`。里面按下面结构放：

```text
private_assets/
  alphapose/
    pretrained_models/             # AlphaPose/HybrIK pose checkpoint
    detector/yolo/data/            # yolov3-spp.weights 等检测器权重
    detector/yolox/data/           # 如果你使用 YOLOX，就放 yolox_l.pth / yolox_x.pth
    model_files/                   # basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

  fromw1/
    H-ACT/retarget/models/smpl/    # SMPL_NEUTRAL.pkl, SMPL_MALE.pkl, SMPL_FEMALE.pkl
    H-ACT/retarget/models/mano/    # MANO_LEFT.pkl, MANO_RIGHT.pkl
    H-ACT/retarget/assets/         # beta, meta, robot/h1 等 retarget assets

  demo_data/
    videos/                        # 如果可以共享，再放原始视频
    smpl_raw/                      # 例如 *_smpl_raw.npy
    fromw1_inputs/                 # 例如 *_623.npy
    fromw1_pkl/                    # 例如 *_623.pkl
    h1_reference/                  # 例如 *_h1_reference_motion.npz
    gifs/                          # 用于快速查看效果
    metrics/                       # 评估报告
```

不要私发或公开发送自己的 API Key。让同门自己创建 `qw_LLM.txt` 或 `.env`。

## 5. 本仓库环境安装

本仓库自身主要依赖 Python 科学计算、可视化和本地前端。前端只用 Python 标准库；H1 GIF 渲染需要 MuJoCo 和 Pillow。

```powershell
cd F:\LLM-pepper\motion_pipeline
conda create -n motion_pipeline python=3.10 -y
conda activate motion_pipeline
pip install numpy scipy pandas pillow imageio opencv-python joblib requests tqdm matplotlib mujoco
```

如果只打开前端查看已有结果，不需要安装 MuJoCo：

```powershell
python frontend\server.py --open
```

或者：

```powershell
run_motion_review_app.bat
```

默认地址是：

```text
http://127.0.0.1:8765
```

## 6. AlphaPose 安装和使用

AlphaPose 建议单独安装在 `F:/LLM-pepper/AlphaPose`。

### 6.1 获取代码

```powershell
cd F:\LLM-pepper
git clone https://github.com/MVIG-SJTU/AlphaPose.git AlphaPose
cd AlphaPose
```

如果你准备直接发给同门你本机修改过的 AlphaPose 代码，也要注意不要把 `pretrained_models/`、`model_files/`、`examples/out/`、`scripts/_623_out/` 等生成数据和模型权重提交到公开 GitHub。

### 6.2 创建环境

AlphaPose 官方 README 推荐 Python 3.7+、PyTorch 1.11+、torchvision 0.12+。Windows 下默认不编译 CUDA extension，因此不支持带 `-dcn` 的模型，建议先使用普通 FastPose/HybrIK 配置。

示例：

```powershell
conda create -n alphapose python=3.8 -y
conda activate alphapose

# 按本机 CUDA 版本安装 PyTorch。下面只是 CUDA 11.3 示例。
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install cython
pip install cython_bbox
pip install pycocotools
python setup.py build develop
```

如果 `pycocotools` 或 `cython_bbox` 在 Windows 编译失败，可以用你已经能运行的环境为准，把对应环境导出给同门：

```powershell
conda env export -n alphapose > alphapose_env.yml
```

### 6.3 准备 AlphaPose 必需文件

至少需要：

```text
AlphaPose/pretrained_models/       pose/SMPL checkpoint
AlphaPose/detector/yolo/data/      yolov3-spp.weights，或者
AlphaPose/detector/yolox/data/     yolox_l.pth / yolox_x.pth
AlphaPose/model_files/             basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

SMPL 模型需要从 SMPL 官方网站申请下载，不能随意公开上传。

### 6.4 AlphaPose 输出约定

每个动作使用稳定的 `motion_id`，例如：

```text
beckon_001
bow_001
point_left_001
wave_right_001
```

建议把原始视频放到：

```text
motion_pipeline/data/videos/<motion_id>.mp4
```

AlphaPose 运行后，把输出整理为：

```text
motion_pipeline/data/alphapose_raw/<motion_id>/alphapose-results.json
motion_pipeline/data/smpl/smpl_raw/<motion_id>_smpl_raw.npy
```

如果继续使用你在 AlphaPose 里改过的 `scripts/video_to_623_npy.py`，同门可以先在 AlphaPose 仓库里跑通，再把 `*_smpl_raw.npy` 复制到本仓库的 `data/smpl/smpl_raw/`。

## 7. FRoM-W1 安装和使用

FRoM-W1 建议单独安装在 `F:/LLM-pepper/FRoM-W1`。本项目目前主要使用 FRoM-W1 的 `H-ACT/retarget` 部分，把 623D 特征转换成 H1 pkl。

### 7.1 获取代码

```powershell
cd F:\LLM-pepper
git clone https://github.com/OpenMOSS/FRoM-W1.git FRoM-W1
cd FRoM-W1
```

### 7.2 创建 retarget 环境

你本机之前使用的是类似 `F:\anaconda\envs\retarget\python.exe` 的环境。给同门复现时建议单独建一个环境：

```powershell
conda create -n retarget python=3.10 -y
conda activate retarget
cd F:\LLM-pepper\FRoM-W1
pip install -r requirements.txt
pip install -r H-ACT\retarget\requirements.txt
```

如果 MANO 依赖安装失败，可以先用：

```powershell
pip install -r H-ACT\retarget\requirements_no_mano.txt
```

但如果要完整处理手部，仍然需要 MANO 相关依赖和模型文件。

### 7.3 准备 FRoM-W1 retarget 必需文件

需要把以下文件放到 FRoM-W1 的 retarget 目录：

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

其中 SMPL/MANO 需要遵守各自官方许可；retarget assets 需要遵守 FRoM-W1/HuggingFace 数据说明。公开 GitHub 不要直接放这些大文件。

### 7.4 本项目调用 FRoM-W1 的方式

本项目不会直接改 FRoM-W1 内部代码，而是通过脚本调用：

```powershell
cd F:\LLM-pepper\motion_pipeline

python adapters\canonical_to_fromw1_623.py `
  --input-dir data\canonical\canonical_edited `
  --glob "*_canonical_motion_v2.npz" `
  --output-dir data\fromw1_inputs\fromw1_inputs_canonical_v2 `
  --fromw1-root F:\LLM-pepper\FRoM-W1
```

然后在 retarget 环境里生成 H1 pkl：

```powershell
cd F:\LLM-pepper\FRoM-W1\H-ACT\retarget

F:\anaconda\envs\retarget\python.exe F:\LLM-pepper\motion_pipeline\tools\fromw1_623_to_pkl_batch.py `
  --input-dir F:\LLM-pepper\motion_pipeline\data\fromw1_inputs\fromw1_inputs_canonical_v2 `
  --output-dir F:\LLM-pepper\motion_pipeline\data\fromw1_pkl\fromw1_pkl_canonical_v2 `
  --fromw1-retarget-root F:\LLM-pepper\FRoM-W1\H-ACT\retarget `
  --robot H1 `
  --hand-type dex3 `
  --output-fps 60
```

再把 pkl 转成本项目统一的 H1 参考动作：

```powershell
cd F:\LLM-pepper\motion_pipeline

python adapters\fromw1_pkl_to_h1_reference.py `
  --input-dir data\fromw1_pkl\fromw1_pkl_canonical_v2 `
  --output-dir data\h1_reference\fromw1_pkl_canonical_v2
```

生成 H1 参考动作 GIF：

```powershell
F:\anaconda\envs\retarget\python.exe tools\h1_reference_to_gif.py `
  --input-dir data\h1_reference\fromw1_pkl_canonical_v2 `
  --output-dir data\gifs\h1_reference_gifs\fromw1_pkl_canonical_v2
```

## 8. 从 SMPL 到 FRoM-W1 的推荐复现流程

假设 `data/smpl/smpl_raw/` 已经有 `*_smpl_raw.npy`：

### 8.1 AlphaPose 层质量评估

```powershell
python scripts\evaluate_alphapose_smpl.py `
  --input-dir data\smpl\smpl_raw `
  --output-dir data\metrics\alphapose_quality
```

### 8.2 修复和压缩 SMPL

```powershell
python scripts\repair_alphapose_smpl_batch.py `
  --input-dir data\smpl\smpl_raw `
  --repaired-dir data\smpl\smpl_repaired `
  --compact-dir data\smpl\smpl_repaired_compact `
  --metrics-dir data\metrics\alphapose_repair
```

### 8.3 生成 canonical motion

```powershell
python scripts\smpl_repaired_to_canonical.py `
  --input-dir data\smpl\smpl_repaired_compact `
  --output-dir data\canonical\canonical_original `
  --fps 30
```

### 8.4 修复 canonical motion

```powershell
python scripts\repair_canonical_motion.py `
  --input-dir data\canonical\canonical_original `
  --glob "*_canonical_motion.npz" `
  --output-dir data\canonical\canonical_edited `
  --suffix "_canonical_motion_v2.npz"
```

### 8.5 生成 FRoM-W1 输入和 H1 pkl

```powershell
python adapters\canonical_to_fromw1_623.py `
  --input-dir data\canonical\canonical_edited `
  --glob "*_canonical_motion_v2.npz" `
  --output-dir data\fromw1_inputs\fromw1_inputs_canonical_v2 `
  --fromw1-root F:\LLM-pepper\FRoM-W1
```

```powershell
cd F:\LLM-pepper\FRoM-W1\H-ACT\retarget

F:\anaconda\envs\retarget\python.exe F:\LLM-pepper\motion_pipeline\tools\fromw1_623_to_pkl_batch.py `
  --input-dir F:\LLM-pepper\motion_pipeline\data\fromw1_inputs\fromw1_inputs_canonical_v2 `
  --output-dir F:\LLM-pepper\motion_pipeline\data\fromw1_pkl\fromw1_pkl_canonical_v2 `
  --fromw1-retarget-root F:\LLM-pepper\FRoM-W1\H-ACT\retarget
```

### 8.6 转 H1 reference 并可视化

```powershell
cd F:\LLM-pepper\motion_pipeline

python adapters\fromw1_pkl_to_h1_reference.py `
  --input-dir data\fromw1_pkl\fromw1_pkl_canonical_v2 `
  --output-dir data\h1_reference\fromw1_pkl_canonical_v2

F:\anaconda\envs\retarget\python.exe tools\h1_reference_to_gif.py `
  --input-dir data\h1_reference\fromw1_pkl_canonical_v2 `
  --output-dir data\gifs\h1_reference_gifs\fromw1_pkl_canonical_v2
```

## 9. 一键脚本

如果已经有一个修改后的 canonical npz，可以使用：

```powershell
scripts\run_reference_to_robojudo.bat data\canonical\canonical_fromw1_llm_edited\point_left_001_canonical_motion_llm_v3.npz point_left_001_llm_v3
```

这个脚本会执行：

```text
canonical npz
  -> FRoM-W1 623 npy
  -> FRoM-W1 H1 pkl
  -> h1_reference_motion.npz
  -> pkl GIF
  -> RoboJuDo rollout data + execution GIF
```

脚本默认环境路径：

```text
PIPELINE_PYTHON=python
RETARGET_PYTHON=F:\anaconda\envs\retarget\python.exe
ROBOJUDO_PYTHON=F:\anaconda\envs\robojudo\python.exe
FROMW1_ROOT=F:\LLM-pepper\FRoM-W1
```

同门机器路径不同的话，可以在 PowerShell 里先设置环境变量，或者直接改 bat 顶部默认值。

## 10. 前端可视化迭代

启动：

```powershell
cd F:\LLM-pepper\motion_pipeline
python frontend\server.py --open
```

前端功能：

```text
选择动作 motion_id
选择版本 original / canonical_v2 / llm_v3
选择后端 FRoM-W1 / ExBody
同时查看原始视频、H1 参考动作 GIF、机器人实际执行 GIF
查看千问视觉模型描述和千问文本模型修改计划
输入用户评价并保存到 data/feedback/
```

目前前端只负责查看和保存用户反馈，不会自动调用千问或自动改动作。

## 11. 千问 API 文件

调用千问脚本时需要 API key。不要把密钥放进 GitHub。

建议让同门自己创建：

```text
F:/LLM-pepper/qw_LLM.txt
```

文件内容只放一行 API key。

相关脚本：

```text
scripts/qwen_vl_video_intent.py
scripts/qwen_vl_robot_visual_edit_suggestions.py
scripts/call_qwen_max_motion_editor.py
```

## 12. 给同门的最小交接建议

最稳的交接方式不是让同门从零跑完整链路，而是分三步：

1. 先让他只跑本仓库前端，查看你私发的一组 demo 数据。
2. 再让他安装 FRoM-W1 retarget，复现 `623 npy -> H1 pkl -> h1_reference GIF`。
3. 最后再安装 AlphaPose，复现 `视频 -> SMPL raw npy`。

这样可以快速定位问题属于：

```text
AlphaPose 环境问题
FRoM-W1 retarget 环境问题
本项目转换脚本问题
RoboJuDo 执行问题
```

## 13. 常见问题

### 13.1 为什么不把 AlphaPose 和 FRoM-W1 直接放进本仓库？

因为它们是第三方项目，体积大、依赖复杂、包含模型权重和许可证边界。本仓库只保存适配层和流程说明更清晰。

### 13.2 为什么不公开上传 SMPL/MANO？

SMPL/MANO 需要用户到官方网站注册并同意许可后下载。公开上传可能违反许可。

### 13.3 为什么生成数据默认不上传？

视频、npy、npz、pkl、gif 通常体积大，而且可能包含人物隐私或本机实验数据。共享给同门可以私发小型 demo 包。

### 13.4 同门只想看效果，需要装 AlphaPose 和 FRoM-W1 吗？

不需要。只要你私发 `data/videos/`、`data/gifs/`、`data/metrics/`、`data/llm/` 中的一组 demo 数据，他运行前端即可查看。

### 13.5 同门想复现 FRoM-W1 pkl 生成，需要 AlphaPose 吗？

不需要。只要你私发或 GitHub 提供 `data/fromw1_inputs/.../*_623.npy`，他只需要装 FRoM-W1 retarget 环境。

### 13.6 同门想从视频开始完整复现，需要什么？

需要：

```text
AlphaPose 代码和环境
AlphaPose detector/pose/SMPL 模型文件
FRoM-W1 代码和 retarget 环境
FRoM-W1 SMPL/MANO/retarget assets
motion_pipeline 本仓库
可共享的输入视频
```

## 14. 发布前检查清单

```powershell
git status --short
```

确认不要出现：

```text
qw_LLM.txt
.env
*.pth, *.pt, *.ckpt, *.bin
SMPL_*.pkl
MANO_*.pkl
basicModel_*.pkl
*.mp4
大量 *.npy, *.npz, *.pkl, *.gif
logs/
```

如果要公开 GitHub，建议 README 里写清楚：

```text
This repository is for academic research and pipeline reproduction.
Third-party repositories, model weights, body models, and private videos are not included.
```
