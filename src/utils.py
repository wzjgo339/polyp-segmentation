"""
通用工具函数
"""
import os
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """
    设置全局随机种子，确保实验可重复
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_mask(mask):
    """
    将 mask 从 {0, 255} 归一化为 {0.0, 1.0}
    输入: (H, W) uint8
    输出: (1, H, W) float32
    """
    mask = mask.astype(np.float32) / 255.0
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def denormalize_mask(mask):
    """
    将 mask 从 [0,1] 取整到 {0, 255} uint8
    """
    return (mask * 255).astype(np.uint8)


def get_device():
    """返回可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """统计模型可训练参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_image_files(dir_path):
    """
    获取目录下所有图片文件，按文件名排序
    支持 .png, .jpg, .jpeg
    """
    extensions = {".png", ".jpg", ".jpeg"}
    files = [
        f for f in os.listdir(dir_path)
        if Path(f).suffix.lower() in extensions
    ]
    files.sort()
    return files
