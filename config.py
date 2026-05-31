"""
息肉分割系统 — 全局配置文件
所有超参数集中管理，方便实验调参
"""
import os
from pathlib import Path

# ========== 路径配置 ==========
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# 训练数据（注意：实际目录名为 "image" 而非 "images"）
TRAIN_IMAGE_DIR = PROJECT_ROOT / "TrainDataset" / "image"
TRAIN_MASK_DIR = PROJECT_ROOT / "TrainDataset" / "masks"

# 测试数据（5 个独立子集）
TEST_ROOT = PROJECT_ROOT / "TestDataset"
TEST_SUBSETS = {
    "CVC-300": TEST_ROOT / "CVC-300",
    "CVC-ClinicDB": TEST_ROOT / "CVC-ClinicDB",
    "CVC-ColonDB": TEST_ROOT / "CVC-ColonDB",
    "ETIS-LaribPolypDB": TEST_ROOT / "ETIS-LaribPolypDB",
    "Kvasir": TEST_ROOT / "Kvasir",
}

# 输出目录
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "runs"
RESULT_DIR = PROJECT_ROOT / "results"

# ========== 数据配置 ==========
INPUT_HEIGHT = 384
INPUT_WIDTH = 384
TRAIN_VAL_SPLIT = 0.9  # 90% 训练, 10% 验证

# ========== 训练超参数 ==========
BATCH_SIZE = 16
NUM_EPOCHS = 150
NUM_WORKERS = 2  # Windows 下 DataLoader workers 数不宜太高
PIN_MEMORY = True

# ========== 优化器 ==========
OPTIMIZER = "AdamW"
LR_ENCODER = 1e-4
LR_DECODER = 1e-3
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)

# ========== 学习率调度 ==========
SCHEDULER = "CosineAnnealingLR"
T_MAX = NUM_EPOCHS
ETA_MIN = 1e-6

# ========== 损失函数 ==========
LOSS = "FocalDice"     # Dice + Focal 组合（针对小息肉/难例，替代原 DiceBCE）
DICE_WEIGHT = 0.5       # Dice 损失权重
BCE_WEIGHT = 0.5        # BCE 损失权重（FocalDice 模式下未使用）
FOCAL_ALPHA = 0.25      # Focal Loss α — 正样本权重
FOCAL_GAMMA = 2.0       # Focal Loss γ — 难例聚焦强度

# ========== 训练技巧 ==========
USE_AMP = True           # 混合精度训练（硬性要求）
GRAD_CLIP_NORM = 1.0     # 梯度裁剪
EARLY_STOP_PATIENCE = 20 # 早停 patience
FREEZE_ENCODER_EPOCHS = 10  # 前 N 个 epoch 冻结 encoder

# ========== Multi-scale 训练 ==========
TRAIN_MULTISCALE = True   # 训练时随机切换 384/512 尺寸
MULTISCALE_SIZES = [(384, 384), (512, 512)]  # 候选训练尺寸

# ========== 模型配置 ==========
ENCODER_NAME = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
MODEL_ARCH = "Unet"   # "Unet" | "UnetPlusPlus" | "DeepLabV3Plus"
MODEL_CLASSES = 1        # 二分类（息肉 / 背景）
ACTIVATION = None        # 训练时无激活（配合 BCEWithLogitsLoss）
DECODER_USE_BATCHNORM = True

# ========== 数据增强 ==========
AUGMENTATION = {
    "random_rotate90_prob": 0.5,
    "horizontal_flip_prob": 0.5,
    "shift_limit": 0.1,
    "scale_limit": 0.2,
    "rotate_limit": 30,
    "shift_scale_rotate_prob": 0.5,
    "elastic_alpha": 1,
    "elastic_sigma": 50,
    "elastic_alpha_affine": 50,
    "elastic_prob": 0.3,
    "brightness_limit": 0.15,
    "contrast_limit": 0.15,
    "brightness_contrast_prob": 0.3,
    "gaussian_blur_limit": (3, 5),
    "gaussian_blur_prob": 0.1,
}

# 归一化（ImageNet 标准）
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# ========== 评估配置 ==========
EVAL_THRESHOLD = 0.5     # 二值化阈值
TTA_ENABLED = True       # 测试时增强（水平翻转平均）
TTA_FLIP_PROB = 0.5

# 高分辨率测试集使用更大的推理尺寸（原图 966×1225 → 384 丢失太多细节）
INFER_SIZE_LARGE = 512
HIGH_RES_DATASETS = ["CVC-ColonDB", "ETIS-LaribPolypDB"]

# ========== 推理配置 ==========
PREDICT_THRESHOLD = 0.5
SAVE_OVERLAY = True       # 保存预测叠加图

# ========== 随机种子 ==========
SEED = 42
