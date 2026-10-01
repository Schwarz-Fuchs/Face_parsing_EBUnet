# Face Parsing：EBUNet

本项目使用 PyTorch 实现 EBUNet 人脸解析模型，对输入人脸图像进行 19 类像素级分割。仓库包含模型结构、数据加载、训练、推理、损失函数和 CelebAMask-HQ 数据预处理脚本，并附有训练检查点与部分预测结果。

## 功能概览

- 输入人脸 RGB 图像，默认缩放至 `512 × 512`。
- 输出单通道类别掩码（每个像素的类别编号）以及彩色可视化掩码。
- 使用 19 类设置：背景加 18 种脸部区域/配件类别。
- 支持训练阶段的数据增强、类别权重、OHEM/Dice/结构一致性相关损失及可选过采样配置。
- 在 `models/EBUnet/` 和 `models/EBUnet_oversamp/` 中提供若干训练步数的检查点。

> 当前训练和测试代码在模型构建及张量处理处直接调用 `.cuda()`，并在 `main.py` 中默认设置 `CUDA_VISIBLE_DEVICES=0`。因此仓库当前流程按 NVIDIA GPU 设计，不能仅靠把设备参数改成 `cpu` 就在 CPU 上运行。

## 目录结构

```text
Face_parsing_EBUnet/
├── main.py                    # 训练/测试程序入口
├── parameter.py               # 命令行参数和默认路径
├── EBUnet.py                  # EBUNet 与增强版 EBUNet_Enhanced 网络定义
├── data_loader.py             # CelebAMaskHQ 数据集读取、增强及 DataLoader
├── trainer.py                 # 训练循环、优化器、损失、验证、日志和检查点保存
├── tester.py                  # 加载检查点并批量输出类别掩码/彩色掩码
├── loss_functions.py          # OHEM、Soft Dice 和结构一致性等损失实现
├── utils.py                   # 调色板、标签转换、文件夹及图像辅助工具
├── requirements.txt           # 当前记录的 Python 依赖版本
├── run.sh                     # 训练命令示例
├── run_test.sh                # 测试/预测命令示例
├── Data_preprocessing/        # CelebAMask-HQ 标签转换、数据划分和着色脚本
├── data/
│   ├── train/images/          # 训练 RGB 人脸图像
│   ├── train/masks/           # 训练单通道类别标签
│   ├── val/images/            # 验证图像
│   ├── test/images/           # 测试图像
│   └── val|test/predicted_*   # 已生成的预测结果
├── models/
│   ├── EBUnet/                 # EBUnet 实验检查点及类别权重缓存
│   └── EBUnet_oversamp/        # 另一组实验检查点及类别权重缓存
├── logs/EBUnet/               # 训练文本日志
├── runs/training/             # TensorBoard event 文件
└── samples/                   # 数据/训练过程检查图像
```

## 代码文件说明

| 文件 | 说明 |
|---|---|
| `main.py` | 根据 `--mode` 选择训练或测试；训练时创建日志、样例和模型目录，初始化数据加载器及 `Trainer`。 |
| `parameter.py` | 集中定义图像大小、训练步数、batch size、学习率、输入输出路径、模型版本及保存/评估频率等 CLI 参数。 |
| `EBUnet.py` | EBUNet 网络结构，含基础及增强版结构、特征提取/融合和解码模块。当前训练和测试都实例化 `EBUNet_Enhanced(classes=19)`。 |
| `data_loader.py` | 读取图像与标签配对，训练时对两者应用一致的几何变换；图像使用双线性类插值、掩码使用最近邻插值，并对水平翻转后的左右脸部类别编号做交换。 |
| `trainer.py` | 模型初始化、AdamW、学习率 warmup/余弦调度、训练与验证、损失计算、TensorBoard 写入和 checkpoint 保存/恢复。 |
| `tester.py` | 从 `models/<version>/<model_name>` 读取检查点，对 `test_image_path` 中图像分批预测，并写出原始类别索引掩码和彩色可视化结果。 |
| `loss_functions.py` | OHEM、Soft Dice、结构一致性等像素级分割损失实现。具体组合和权重在 `trainer.py` 中配置。 |
| `utils.py` | CelebAMask-HQ 调色板、预测 logits 转类别图/彩色图、标签辅助函数等。 |
| `Data_preprocessing/g_mask.py` | 将 CelebAMask-HQ 按部件保存的掩码合并为单张类别索引掩码；输入目录和图片数量在脚本顶部固定。 |
| `Data_preprocessing/g_partition.py` | 依据 CelebA-HQ mapping 文件划分并复制原始图像和掩码；源目录、目标目录及拆分逻辑写在脚本内。 |
| `Data_preprocessing/g_color.py` | 将类别索引掩码转换成彩色图；类别颜色表、路径和处理数量写在脚本内。 |
| `run.sh`、`run_test.sh` | 训练和预测的 shell 命令示例。实际运行前应核对参数、路径、模型版本目录及 CUDA 配置。 |

## 环境准备

仓库 `requirements.txt` 锁定了特定版本，例如 `torch==2.8.0+cu129`，且同时列出 `opencv-python`、`opencv-contrib-python` 和 `opencv-python-headless`；还重复声明了两个不同版本的 Pillow。这些组合在不同平台上可能无法直接解析或安装，建议根据目标机器新建隔离环境，并按其 NVIDIA 驱动/CUDA 兼容性选择 PyTorch，再安装项目实际需要的依赖：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

至少需要 PyTorch、torchvision、NumPy、Pillow、OpenCV、SciPy、pandas、TensorBoardX 和 tqdm。具体版本需要在目标系统上验证。尽量不要在同一环境同时安装多个 OpenCV wheel；通常保留 `opencv-python` 即可，若需要无 GUI 环境可选 `opencv-python-headless`。清理依赖清单时应只保留一个 Pillow 版本，并确认 NumPy 与 pandas/SciPy 的二进制兼容性。

检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

## 数据准备

### 训练数据目录约定

训练 `Data_Loader` 需要图像和索引掩码分别放在以下目录中，并通过文件名配对：

```text
data/
├── train/
│   ├── images/   # .jpg 或 .png 的 RGB 图像
│   └── masks/    # 对应文件名的单通道类别索引掩码 PNG
├── val/
│   ├── images/
│   └── masks/    # 若启用验证评估，应准备对应 ground-truth 掩码
└── test/
    └── images/
```

当前仓库的 `data/val/` 和 `data/test/` 下可以看到 `predicted_masks`、`predicted_masks_color` 预测输出目录；这些是推理结果目录，不是验证/测试真值掩码目录。重新训练或评估前请确认相应的 ground-truth 标签已准备，并且训练代码实际读取的配置路径指向真值标签。

图像与掩码须同名（仅扩展名可由 `.jpg` 对应 `.png`），尺寸和空间位置对应。掩码应该是类别 ID 的单通道图，而不是 RGB 彩色预览图。当前模型 `classes=19`，有效类别编号应与 `utils.py` 调色板及 `data_loader.py` 中左右翻转类别映射一致。不要对标签使用双线性插值，否则会产生非法类别 ID。

### CelebAMask-HQ 预处理脚本

`Data_preprocessing/` 中脚本针对 CelebAMask-HQ 原始目录布局编写，且含有硬编码的文件夹名、图片数量和输出路径。`g_mask.py` 将部件掩码合成为类别索引掩码；`g_partition.py` 依赖 CelebA-HQ mapping 文件划分数据；`g_color.py` 生成彩色掩码示例。运行前应先按原始数据集实际目录修改脚本配置，检查数据集许可，并避免覆盖已有数据。

## 训练

训练参数由 `parameter.py` 定义。默认训练数据为 `./data/train/images` 和 `./data/train/masks`；默认输入尺寸 512、总步数 20,000、batch size 64。考虑显存，请按机器情况调小 batch size。一个更贴近当前仓库目录的示例：

```bash
python main.py \
  --mode train \
  --version EBUnet \
  --batch_size 8 \
  --imsize 512 \
  --total_step 20000 \
  --img_path ./data/train/images \
  --label_path ./data/train/masks \
  --test_image_path ./data/val/images \
  --test_label_path ./data/val/masks
```

`--version` 决定检查点、日志和训练样例所在的子目录：模型写入 `models/<version>/`，文本日志写入 `logs/<version>/`，训练样例写入 `samples/<version>/`。训练过程会按 `--model_save_step` 保存检查点，默认每 5,000 步一次；默认 warmup 500 步、学习率 `5e-4`，并按配置使用余弦退火。

可选过采样参数为 `--over_sampler True`。`main.py` 会基于训练掩码统计指定类别并构造 rare-class list；传入该参数前确认数据类别 ID 和过采样逻辑符合当前数据集。仓库 `run.sh` 将版本命名为 `EBUnet_oversamp`，但命令本身没有传 `--over_sampler True`，因此名称不能证明那组权重确实使用了过采样。

恢复训练可设置 `--pretrained_model` 为要恢复的步数，例如：

```bash
python main.py --mode train --version EBUnet --pretrained_model 20000
```

恢复代码会从 `models/<version>/checkpoint_<步数>_steps.pth` 加载模型和优化器/调度器状态。启动前请确认该文件存在。

## 测试与预测

测试模式使用 `--version` 和 `--model_name` 共同确定 checkpoint 路径：

```text
models/<version>/<model_name>
```

例如，使用仓库已有的 `models/EBUnet/checkpoint_20000_steps.pth` 对测试图像预测：

```bash
python main.py \
  --mode test \
  --version EBUnet \
  --model_name checkpoint_20000_steps.pth \
  --batch_size 8 \
  --imsize 512 \
  --test_image_path ./data/test/images \
  --test_label_path ./data/test/predicted_masks \
  --test_color_label_path ./data/test/predicted_masks_color
```

预测结果：

- `predicted_masks/`：单通道类别索引图，像素值为 0–18，可用于后续程序处理。
- `predicted_masks_color/`：按调色板着色的可视化结果，便于人工查看。

## 检查点与日志

每个 `models/<version>/` 下的 `checkpoint_<step>_steps.pth` 保存训练状态；目录中的 `median_weights.npy` 是按训练标签统计/缓存的类别权重。仓库目前同时包含 `EBUnet` 和 `EBUnet_oversamp` 两套实验检查点，使用时须根据实际实验配置选择。

训练文本日志位于 `logs/<version>/log.txt`，TensorBoard event 写入 `runs/training/`。查看 TensorBoard：

```bash
tensorboard --logdir runs
```

