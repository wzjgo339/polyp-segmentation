#!/usr/bin/env python
"""
肠息肉分割系统 — 评估入口

用法：
    conda activate myPytorch
    python evaluate.py

功能：
    - 在 5 个测试集上分别评估最佳模型
    - 输出 Dice、IoU、Precision、Recall 表格
    - 生成可视化对比图
    - 结果保存至 results/summary.csv
"""
import sys
sys.path.insert(0, '.')

from src.evaluate import run_evaluation


def main():
    print("=" * 55)
    print("  肠息肉分割系统 — 跨数据集评估")
    print("  Model: U-Net + EfficientNet-B4")
    print("=" * 55)

    run_evaluation()


if __name__ == "__main__":
    main()
