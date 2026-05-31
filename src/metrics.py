"""
评估指标计算 — Dice, IoU, Precision, Recall

注意：mask 值域为 [0, 1]（float），预测经 sigmoid + 阈值化后比较
"""
import torch
from config import EVAL_THRESHOLD


def dice_coeff(pred_logits, target, threshold=EVAL_THRESHOLD, smooth=1e-5):
    """
    Dice Coefficient (F1 Score for segmentation)

    Args:
        pred_logits: (B, 1, H, W) — 模型 logits
        target:      (B, 1, H, W) — GT mask [0, 1]
        threshold:   二值化阈值
        smooth:      epsilon 避免除零
    Returns:
        dice (float): 批量平均 Dice
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()

    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(pred_logits, target, threshold=EVAL_THRESHOLD, smooth=1e-5):
    """
    IoU (Jaccard Index)

    Args:
        pred_logits: (B, 1, H, W)
        target:      (B, 1, H, W)
        threshold:   二值化阈值
    Returns:
        iou (float)
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()

    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def precision_recall(pred_logits, target, threshold=EVAL_THRESHOLD, smooth=1e-5):
    """
    计算 Precision 和 Recall

    Returns:
        (precision, recall)
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()

    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    tp = (pred * target).sum(dim=1)
    fp = pred.sum(dim=1) - tp
    fn = target.sum(dim=1) - tp

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return precision.mean().item(), recall.mean().item()


def compute_all_metrics(pred_logits, target, threshold=EVAL_THRESHOLD):
    """
    一键计算所有指标

    Returns:
        dict: {"dice": ..., "iou": ..., "precision": ..., "recall": ...}
    """
    dice = dice_coeff(pred_logits, target, threshold)
    iou = iou_score(pred_logits, target, threshold)
    prec, rec = precision_recall(pred_logits, target, threshold)

    return {
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
    }
