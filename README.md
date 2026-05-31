# 肠息肉图像分割系统

基于深度学习的结肠镜息肉语义分割系统，使用 **U-Net + EfficientNet-B4** 架构，在 Kvasir-SEG 和 CVC-ClinicDB 混合数据集上训练，并在五个独立测试集上评估跨数据集泛化性能。

---

## 项目结构

```
D:\CV_deepLearning\息肉分割\
├── TrainDataset/           # 训练数据（1,450 张）
│   ├── image/              # 结肠镜 RGB 图像
│   └── masks/              # 二值分割 mask（{0, 255}）
├── TestDataset/            # 测试数据（5 个子集，798 张）
│   ├── CVC-300/            # 60 张
│   ├── CVC-ClinicDB/       # 62 张
│   ├── CVC-ColonDB/        # 380 张
│   ├── ETIS-LaribPolypDB/  # 196 张（最高分辨率 966×1225）
│   └── Kvasir/             # 100 张
│
├── src/                    # 源代码
│   ├── dataset.py          # 数据集类 + 数据增强管线
│   ├── model.py            # U-Net + EfficientNet-B4 模型定义
│   ├── train.py            # 训练引擎（AMP / 早停 / TensorBoard）
│   ├── evaluate.py         # 跨数据集评估（5 测试集独立报告）
│   ├── predict.py          # 单张图像推理 + 可视化
│   ├── metrics.py          # Dice / IoU / Precision / Recall
│   ├── loss.py             # DiceLoss + BCE 组合损失
│   └── utils.py            # 工具函数
│
├── config.py               # 全局超参数配置
├── train.py                # 训练入口
├── evaluate.py             # 评估入口
├── predict.py              # 推理入口
├── checkpoints/            # 模型权重保存
├── runs/                   # TensorBoard 日志
├── results/                # 评估结果 + 可视化图
│
├── README.md               # 本文件
└── prompt.md               # 原始需求文档
```

---

## 环境配置

### 前置条件

| 项目 | 规格 |
|------|------|
| Conda 环境 | `myPytorch` |
| GPU | NVIDIA RTX 5060 Ti (16GB VRAM) |
| PyTorch | 2.9.0.dev20250825+cu128 |
| CUDA | 12.8 |

### 激活环境

```bash
conda activate myPytorch
```

### 依赖安装（如缺包）

```bash
pip install segmentation_models_pytorch albumentations tensorboardX
```

---

## 数据集

### 结构

数据集已就绪，无需额外下载。所有图像为 RGB PNG，mask 为单通道 PNG（像素值 {0, 255}）。

| 分割 | 子集 | 样本数 | 分辨率 |
|------|------|--------|--------|
| 训练 | TrainDataset (Kvasir-SEG + CVC-ClinicDB) | 1,450 | 多样 (288×384 ~ 1070×1348) |
| 验证 | 从 TrainDataset 9:1 随机划分 | 145 | — |
| 测试 | CVC-300 | 60 | 500×574 |
| 测试 | CVC-ClinicDB | 62 | 288×384 |
| 测试 | CVC-ColonDB | 380 | 500×574 |
| 测试 | ETIS-LaribPolypDB | 196 | 966×1225 |
| 测试 | Kvasir | 100 | 多样 |

### 注意事项

- TrainDataset 的图片目录名为 **image**（单数），非 images（复数）
- Mask 像素值为 {0, 255}，代码中自动归一化为 {0.0, 1.0}
- 息肉平均占比仅 13.11%（类别严重不平衡）

---

## 训练

### 启动训练

```bash
conda activate myPytorch
cd D:\CV_deepLearning\息肉分割
python train.py
```

### 训练策略

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | U-Net + EfficientNet-B4 | 预训练 encoder (ImageNet) |
| 输入分辨率 | 384 × 384 | 全部 resize 到统一尺寸 |
| Batch size | 16 | 适配 16GB VRAM |
| 优化器 | AdamW | 差分学习率（enc: 1e-4 / dec: 1e-3） |
| 学习率调度 | CosineAnnealingLR | T_max=150, eta_min=1e-6 |
| 损失函数 | Dice + BCE (1:1) | 抗类别不平衡 |
| 混合精度 | FP16 (AMP) | 显存减半，速度提升 |
| 早停 | patience=20 | 监控验证集 Dice |
| Encoder 冻结 | 前 10 个 epoch | 保护预训练特征 |
| 数据增强 | 翻转/旋转/弹性变换/色彩抖动 | albumentations |

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir runs/
```

浏览器打开 `http://localhost:6006` 查看 loss、Dice、IoU 曲线。

### 输出

- **最佳模型**: `checkpoints/best_model.pth`
- **训练日志**: `runs/` (TensorBoard 格式)

---

## 评估

### 启动评估

```bash
conda activate myPytorch
cd D:\CV_deepLearning\息肉分割
python evaluate.py
```

### 评估指标

在全部 5 个测试集上**独立**报告以下指标：

- **Dice Coefficient**（主指标）
- **IoU (Jaccard Index)**
- **Precision / Recall**

### 输出示例

```
=======================================================
  评估结果汇总
=======================================================
Dataset              Dice     IoU      Prec     Recall
-------------------------------------------------------
CVC-300              0.xxxx   0.xxxx   0.xxxx   0.xxxx
CVC-ClinicDB         0.xxxx   0.xxxx   0.xxxx   0.xxxx
CVC-ColonDB          0.xxxx   0.xxxx   0.xxxx   0.xxxx
ETIS-LaribPolypDB    0.xxxx   0.xxxx   0.xxxx   0.xxxx
Kvasir               0.xxxx   0.xxxx   0.xxxx   0.xxxx
-------------------------------------------------------
Mean                 0.xxxx   0.xxxx
Std                  0.xxxx   0.xxxx
=======================================================
```

### 输出文件

- `results/summary.csv` — 数值结果 CSV
- `results/<subset_name>/` — 每个测试集 10 张可视化对比图

---

## 推理

### 单张图像推理

```bash
# 基础推理
python predict.py --image 图片路径.png

# 启用 TTA（测试时增强）
python predict.py --image 图片路径.png --tta

# 指定模型权重
python predict.py --image 图片路径.png --weights checkpoints/best_model.pth

# 指定输出目录
python predict.py --image 图片路径.png --output results/my_pred
```

### 输出

推理生成三栏对比图（并排显示）：

| 左栏 | 中栏 | 右栏 |
|------|------|------|
| 原始图像 | 二值分割 Mask | 预测 Mask 叠加（绿色） |

输出文件：
- `results/predict/<文件名>_result.jpg` — 三栏可视化图
- `results/predict/<文件名>_mask.png` — 二值分割 mask

---

## 模型架构

```
Input (3, 384, 384)
    │
    ▼
EfficientNet-B4 Encoder (ImageNet pretrained)
    │
    ├── Stage 1: (64, 192, 192)   → Decoder
    ├── Stage 2: (256, 96, 96)    → Decoder
    ├── Stage 3: (512, 48, 48)    → Decoder
    └── Stage 4: (1024, 24, 24)   → Decoder
    │
    ▼
U-Net Decoder (上采样 + 跳跃连接)
    │
    ▼
Output (1, 384, 384) → Sigmoid → 二值 Mask
```

- 总参数量: **20.23M**
- 预训练权重: ImageNet (via segmentation_models_pytorch)

---

## 损失函数

**组合损失**: `DiceLoss + BCEWithLogitsLoss (1:1)`

- **Dice Loss**: 直接优化 Dice 指标，对前背景不均衡鲁棒
- **BCE Loss**: 稳定早期梯度，加速收敛

---

## 数据增强

使用 albumentations 实现的医学图像专有增强：

| 增强 | 概率 | 说明 |
|------|------|------|
| RandomRotate90 | 0.5 | 旋转增强 |
| HorizontalFlip | 0.5 | 水平翻转 |
| ShiftScaleRotate | 0.5 | 平移/缩放/旋转 |
| ElasticTransform | 0.3 | 弹性变形（模拟组织变形） |
| RandomBrightnessContrast | 0.3 | 亮度/对比度调整 |
| GaussianBlur | 0.1 | 轻度高斯模糊 |

注意：不使用 VerticalFlip（内窥镜图像有方向性）。

---

## 风险与限制

| 风险 | 说明 |
|------|------|
| 跨数据集泛化 | 训练集为 Kvasir + ClinicDB，ETIS/ColonDB 域差异大可能导致指标下降 |
| 类别不均衡 | 息肉仅占 13%，Dice Loss 已提供较好鲁棒性 |
| 过拟合 | 仅 1,450 张训练图，早停 + 数据增强 + ImageNet 预训练已做防护 |
| ETIS 高分辨率 | 966×1225 评估时 resize 为 384×384，可能会丢失细节 |

---

## 参考文献

- Dong et al., "Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers", AIR 2023
- Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MMM 2020
- Bernal et al., "CVC-ClinicDB: A Polyp Image Database", CMIG 2015
- Jakubovskis, "segmentation_models_pytorch" (smp), https://github.com/qubvel/segmentation_models.pytorch
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
