"""
评估引擎 — 在 5 个独立测试集上分别评估模型

功能：
- 加载 best checkpoint
- 分别评估 CVC-300 / CVC-ClinicDB / CVC-ColonDB / ETIS-LaribPolypDB / Kvasir
- 每个测试集报告 Dice、IoU、Precision、Recall
- 表格输出 + CSV 保存
- 可视化对比图（原图 / GT / 预测叠加）
"""
import csv
import time
from PIL import Image as PILImage
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    CHECKPOINT_DIR, RESULT_DIR, TEST_SUBSETS,
    INPUT_HEIGHT, INPUT_WIDTH,
    EVAL_THRESHOLD, NORMALIZE_MEAN, NORMALIZE_STD,
    TTA_ENABLED, INFER_SIZE_LARGE, HIGH_RES_DATASETS,
)
from src.model import create_model
from src.dataset import PolypDataset
from src.metrics import compute_all_metrics
from src.utils import set_seed, ensure_dir, denormalize_mask
from torch.utils.data import DataLoader


def denormalize_image(image_tensor):
    """
    将归一化的图像 tensor 还原为 uint8 RGB 图像
    输入: (C, H, W) float32, ImageNet 归一化
    输出: (H, W, 3) uint8
    """
    mean = np.array(NORMALIZE_MEAN).reshape(3, 1, 1)
    std = np.array(NORMALIZE_STD).reshape(3, 1, 1)

    img = image_tensor.cpu().numpy()
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    return img


def visualize_prediction(image, mask_gt, mask_pred, save_path):
    """
    创建三栏对比图: 原图 | GT Mask | 预测叠加

    Args:
        image:    (H, W, 3) uint8 RGB
        mask_gt:  (H, W) float32 [0, 1]
        mask_pred: (H, W) float32 [0, 1]
        save_path: 保存路径
    """
    # GT mask: 二值化到 {0, 255}
    gt_bin = (mask_gt > EVAL_THRESHOLD).astype(np.uint8) * 255
    pred_bin = (mask_pred > EVAL_THRESHOLD).astype(np.uint8) * 255

    # GT 叠加：红色半透明
    gt_overlay = image.copy()
    gt_overlay[gt_bin > 0] = (255, 0, 0)  # 红色

    # 预测叠加：绿色半透明
    pred_overlay = image.copy()
    pred_overlay[pred_bin > 0] = (0, 255, 0)  # 绿色

    # 三栏并排
    h, w = image.shape[:2]
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = image
    vis[:, w:2*w] = gt_overlay
    vis[:, 2*w:3*w] = pred_overlay

    # 加标签
    label_y = 30
    cv2.putText(vis, "Input", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(vis, "GT (Red)", (w + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(vis, "Pred (Green)", (2 * w + 10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    ret = cv2.imwrite(str(save_path), vis_bgr)
    if not ret:
        PILImage.fromarray(vis).save(str(save_path))


@torch.no_grad()
def evaluate_subset(model, loader, device, subset_name, num_viz=10):
    """
    在单个测试子集上评估

    Args:
        model:        分割模型
        loader:       测试 DataLoader
        device:       torch device
        subset_name:  子集名（用于日志和保存）
        num_viz:      保存的可视化图数量
    Returns:
        metrics (dict): {"dice": ..., "iou": ..., "precision": ..., "recall": ...}
    """
    model.eval()

    all_dice = []
    all_iou = []
    all_prec = []
    all_rec = []

    viz_count = 0
    viz_dir = RESULT_DIR / subset_name
    ensure_dir(viz_dir)

    pbar = tqdm(loader, desc=f"Eval [{subset_name}]")
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        names = batch["name"]

        # 推理
        logits = model(images)

        # TTA（可选）
        if TTA_ENABLED:
            flipped = torch.flip(images, dims=[3])
            logits_flipped = torch.flip(model(flipped), dims=[3])
            logits = (logits + logits_flipped) / 2.0

        # 计算指标
        for i in range(images.size(0)):
            pred_i = logits[i:i+1]
            mask_i = masks[i:i+1]

            metrics = compute_all_metrics(pred_i, mask_i, threshold=EVAL_THRESHOLD)
            all_dice.append(metrics["dice"])
            all_iou.append(metrics["iou"])
            all_prec.append(metrics["precision"])
            all_rec.append(metrics["recall"])

            # 保存可视化
            if viz_count < num_viz:
                img_denorm = denormalize_image(images[i])
                mask_np = masks[i, 0].cpu().numpy()
                pred_prob = torch.sigmoid(pred_i[0, 0]).cpu().numpy()

                base_name = Path(names[i]).stem
                save_path = viz_dir / f"{viz_count}_{base_name}.jpg"
                visualize_prediction(img_denorm, mask_np, pred_prob, save_path)
                viz_count += 1

    # 汇总
    avg_metrics = {
        "dice": float(np.mean(all_dice)),
        "iou": float(np.mean(all_iou)),
        "precision": float(np.mean(all_prec)),
        "recall": float(np.mean(all_rec)),
    }

    return avg_metrics


def run_evaluation():
    """
    完整评估流程：在全部 5 个测试集上评估并报告结果
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(RESULT_DIR)

    print("=" * 55)
    print("  肠息肉分割系统 — 跨数据集评估")
    print(f"  设备: {device}")
    print(f"  TTA: {TTA_ENABLED}")
    print("=" * 55)

    # 加载模型
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"未找到 checkpoint: {checkpoint_path}\n"
            f"请先运行 python train.py 训练模型"
        )

    model = create_model()
    model = model.to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_dice = checkpoint.get("best_dice", "N/A")
    print(f"[Eval] 已加载模型: {checkpoint_path}")
    print(f"[Eval] 训练时最佳验证 Dice: {best_dice}\n")

    # 逐子集评估
    all_results = {}

    for subset_name in TEST_SUBSETS:
        print(f"\n{'='*55}")
        print(f"  评估: {subset_name}")

        # 高分辨率数据集使用更大的推理尺寸
        if subset_name in HIGH_RES_DATASETS:
            infer_h = infer_w = INFER_SIZE_LARGE
            print(f"  高分辨率推理模式: {infer_h}×{infer_w}")
        else:
            infer_h, infer_w = INPUT_HEIGHT, INPUT_WIDTH

        # 构建 DataLoader（使用子集专属尺寸）
        subset_dir = TEST_SUBSETS[subset_name]
        image_dir = subset_dir / "images"
        mask_dir = subset_dir / "masks"

        val_transform = A.Compose([
            A.Resize(height=infer_h, width=infer_w),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2(),
        ])

        dataset = PolypDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            mode="test",
            transform=val_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        print(f"  样本数: {len(dataset)}")

        # 评估
        start_time = time.time()
        metrics = evaluate_subset(model, loader, device, subset_name, num_viz=10)
        elapsed = time.time() - start_time

        all_results[subset_name] = metrics

        print(f"  Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f} | "
              f"耗时: {elapsed:.1f}s")

    # 汇总表格
    print("\n")
    print("=" * 55)
    print("  评估结果汇总")
    print("=" * 55)
    print(f"{'Dataset':<20} {'Dice':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 55)

    dice_list = []
    iou_list = []
    for name in TEST_SUBSETS:
        m = all_results[name]
        dice_list.append(m["dice"])
        iou_list.append(m["iou"])
        print(f"{name:<20} {m['dice']:<8.4f} {m['iou']:<8.4f} "
              f"{m['precision']:<8.4f} {m['recall']:<8.4f}")

    print("-" * 55)
    print(f"{'Mean':<20} {np.mean(dice_list):<8.4f} {np.mean(iou_list):<8.4f}")
    print(f"{'Std':<20} {np.std(dice_list):<8.4f} {np.std(iou_list):<8.4f}")
    print("=" * 55)

    # 保存 CSV
    csv_path = RESULT_DIR / "summary.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "dice", "iou", "precision", "recall"])
        for name in TEST_SUBSETS:
            m = all_results[name]
            writer.writerow([name, f"{m['dice']:.4f}", f"{m['iou']:.4f}",
                            f"{m['precision']:.4f}", f"{m['recall']:.4f}"])
        writer.writerow(["Mean", f"{np.mean(dice_list):.4f}", f"{np.mean(iou_list):.4f}",
                        "", ""])
        writer.writerow(["Std", f"{np.std(dice_list):.4f}", f"{np.std(iou_list):.4f}",
                        "", ""])

    print(f"\n[Eval] 结果已保存: {csv_path}")
    print(f"[Eval] 可视化图已保存至: {RESULT_DIR}/<subset_name>/")

    return all_results
