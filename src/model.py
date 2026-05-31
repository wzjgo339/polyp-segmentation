"""
模型定义 — U-Net + EfficientNet-B4 (via segmentation_models_pytorch)

Encoder: EfficientNet-B4 预训练 (ImageNet)
Decoder: U-Net decoder 上采样
"""
import segmentation_models_pytorch as smp
import torch.nn as nn

from config import (
    ENCODER_NAME, ENCODER_WEIGHTS,
    MODEL_CLASSES, ACTIVATION,
    DECODER_USE_BATCHNORM,
    MODEL_ARCH,
    LR_ENCODER, LR_DECODER,
)
from src.utils import count_parameters


# smp 架构映射
ARCH_MAP = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
}


def create_model():
    """
    创建分割模型（支持多种架构）

    Returns:
        model (nn.Module)
    """
    arch_class = ARCH_MAP.get(MODEL_ARCH, smp.Unet)
    model = arch_class(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=MODEL_CLASSES,
        activation=ACTIVATION,  # None → 配合 BCEWithLogitsLoss
        decoder_use_batchnorm=DECODER_USE_BATCHNORM,
    )

    total, trainable = count_parameters(model)
    print(f"[Model] {MODEL_ARCH} + {ENCODER_NAME}")
    print(f"        总参数量: {total / 1e6:.2f}M")
    print(f"        可训练参数量: {trainable / 1e6:.2f}M")

    return model


def get_optimizer_params(model):
    """
    返回分组后的优化器参数：
    - encoder: 学习率 LR_ENCODER, weight_decay 正常
    - decoder: 学习率 LR_DECODER, weight_decay 正常

    适用于 AdamW 等优化器
    """
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    params_group = [
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": decoder_params, "lr": LR_DECODER},
    ]

    print(f"[Optimizer] Encoder params: {len(encoder_params)} groups, lr={LR_ENCODER}")
    print(f"[Optimizer] Decoder params: {len(decoder_params)} groups, lr={LR_DECODER}")

    return params_group
