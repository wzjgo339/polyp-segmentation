#!/usr/bin/env python
"""
肠息肉分割系统 — 单张图像推理入口

用法：
    conda activate myPytorch
    python predict.py --image 图片路径 [--weights checkpoints/best_model.pth] [--tta] [--output 输出目录]

示例：
    python predict.py --image TestDataset/CVC-300/images/1.png
    python predict.py --image 任意图片.png --tta
    python predict.py --image 图片.png --weights checkpoints/best_model.pth
"""
import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path

from src.predict import predict_image


def main():
    parser = argparse.ArgumentParser(
        description="肠息肉分割 — 单张图像推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True,
                        help="输入图像路径 (支持 PNG/JPG)")
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径 (默认: checkpoints/best_model.pth)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: results/predict)")
    parser.add_argument("--tta", action="store_true",
                        help="启用测试时增强 (HorizontalFlip)")

    args = parser.parse_args()

    # 检查输入文件
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] 文件不存在: {image_path}")
        sys.exit(1)

    # 运行推理
    predict_image(
        image_path=image_path,
        checkpoint_path=args.weights,
        output_dir=Path(args.output) if args.output else None,
        tta=args.tta,
    )


if __name__ == "__main__":
    main()
