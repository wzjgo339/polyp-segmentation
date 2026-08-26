#!/usr/bin/env python
"""
肠息肉分割系统 — 训练入口

用法：
    conda activate myPytorch
    python train.py

功能：
    - 加载 TrainDataset（9:1 训练/验证分割）
    - 训练 U-Net + EfficientNet-B4 模型
    - 混合精度训练 (AMP)
    - TensorBoard 日志记录
    - 自动保存最佳 checkpoint
    - 早停机制
"""
import sys
import argparse
sys.path.insert(0, '.')

from src.dataset import create_dataloaders
from src.train import run_training
from src.utils import set_seed
from config import SEED


def main():
    parser = argparse.ArgumentParser(description="训练肠息肉分割模型")
    parser.add_argument(
        "--resume", action="store_true",
        help="从 checkpoints/best_model.pth 继续训练",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  肠息肉分割系统 — 训练")
    print("  Model: U-Net + EfficientNet-B4")
    print("=" * 50)

    # 设置随机种子
    set_seed(SEED)

    # 创建数据加载器
    print("\n[Data] 加载训练数据...")
    train_loader, val_loader, train_dataset = create_dataloaders()
    print(f"[Data] 训练: {len(train_loader.dataset)} 张")
    print(f"[Data] 验证: {len(val_loader.dataset)} 张")
    if hasattr(train_dataset, 'ms_transforms'):
        sizes = list(train_dataset.ms_transforms.keys())
        print(f"[Data] Multi-scale 训练: {sizes}")

    # 启动训练
    model, best_epoch = run_training(
        train_loader, val_loader, train_dataset, resume=args.resume
    )

    print(f"\n训练完成! 最佳模型保存在 checkpoints/best_model.pth")
    print(f"最佳验证 Dice: epoch {best_epoch}")


if __name__ == "__main__":
    main()
