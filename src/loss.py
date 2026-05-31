"""
损失函数定义 — DiceLoss + BCEWithLogitsLoss 组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DICE_WEIGHT, BCE_WEIGHT, FOCAL_ALPHA, FOCAL_GAMMA


class DiceLoss(nn.Module):
    """
    Dice Loss: 直接优化 Dice Coefficient
    适用于语义分割中的类别不平衡问题

    loss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, 1, H, W) — 模型原始输出（logits）
            target:      (B, 1, H, W) — GT mask, 值域 [0, 1]
        Returns:
            dice_loss (scalar)
        """
        pred = torch.sigmoid(pred_logits)

        # 展平
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Dice Loss + BCE Loss 组合损失

    医学分割常用组合：
    - Dice Loss: 处理类别不平衡，直接优化分割质量
    - BCE Loss: 稳定早期梯度，加速收敛
    """

    def __init__(self, dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target):
        dice = self.dice_loss(pred_logits, target)
        bce = self.bce_loss(pred_logits, target)
        loss = self.dice_weight * dice + self.bce_weight * bce
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss: 进一步压降低置信度背景假阳性
    适用于极端类别不平衡场景

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        # BCEWithLogits 是 AMP-safe 的（内部融合 sigmoid + BCE）
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none'
        )
        # 用 sigmoid 概率计算 focal 权重
        pred = torch.sigmoid(pred_logits)
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        # alpha balancing
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * ce_loss

        return loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Dice Loss + Focal Loss 组合损失

    针对小息肉 / 难例样本优化：
    - Dice Loss: 处理类别不平衡，直接优化分割质量
    - Focal Loss: 聚焦于难分类像素（小息肉边界），降低易分类背景权重
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5,
                 alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, pred_logits, target):
        dice = self.dice_loss(pred_logits, target)
        focal = self.focal_loss(pred_logits, target)
        loss = self.dice_weight * dice + self.focal_weight * focal
        return loss


def get_loss_fn(name="DiceBCE"):
    """
    根据名称返回损失函数

    Args:
        name: "DiceBCE" | "DiceOnly" | "FocalDice"
    Returns:
        loss_fn (nn.Module)
    """
    if name == "DiceBCE":
        return DiceBCELoss()
    elif name == "DiceOnly":
        return DiceLoss()
    elif name == "FocalDice":
        return DiceFocalLoss(dice_weight=0.5, focal_weight=0.5)
    else:
        raise ValueError(f"未知损失函数: {name}")
