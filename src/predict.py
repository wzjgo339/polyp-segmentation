"""
单张图像推理 + 可视化

功能：
- 加载预训练模型
- 推理单张结肠镜图像
- 输出二值分割 mask + 原图叠加可视化
- 支持 TTA
"""
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    INPUT_HEIGHT, INPUT_WIDTH,
    NORMALIZE_MEAN, NORMALIZE_STD,
    PREDICT_THRESHOLD, CHECKPOINT_DIR,
)
from src.model import create_model
from src.utils import ensure_dir, imread_unicode, imwrite_unicode


def get_predict_transform():
    """推理预处理（与训练一致的归一化）"""
    return A.Compose([
        A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def load_model_for_inference(checkpoint_path=None):
    """
    加载训练好的模型用于推理

    Args:
        checkpoint_path: 模型权重路径，默认 checkpoints/best_model.pth
    Returns:
        model, device
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[Predict] 已加载模型: {checkpoint_path}")
    print(f"[Predict] 设备: {device}")

    return model, device


@torch.no_grad()
def predict_single(model, device, image, transform, tta=False):
    """
    单张图像推理

    Args:
        model:     分割模型
        device:    torch device
        image:     (H, W, 3) uint8 RGB 图像
        transform: 预处理变换
        tta:       是否启用 TTA (HorizontalFlip)
    Returns:
        mask_prob:  (H, W) float32 [0, 1] — 归一化后的预测概率图 (原始分辨率)
        mask_bin:   (H_info, W_info) uint8 {0, 255} — 二值化 mask (原始分辨率)
    """
    original_h, original_w = image.shape[:2]

    # 预处理
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    # 推理
    logits = model(input_tensor)

    if tta:
        flipped = torch.flip(input_tensor, dims=[3])
        logits_flipped = torch.flip(model(flipped), dims=[3])
        logits = (logits + logits_flipped) / 2.0

    # 输出概率图 (resized size)
    prob = torch.sigmoid(logits[0, 0]).cpu().numpy()  # (H, W) float32

    # 恢复到原始分辨率
    prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    mask_bin = (prob > PREDICT_THRESHOLD).astype(np.uint8) * 255

    return prob, mask_bin


def visualize_prediction(image, mask_bin, prob_map, save_path=None):
    """
    创建可视化：原图 / 二值 mask / 叠加图（三栏并排）

    Args:
        image:    (H, W, 3) uint8 RGB
        mask_bin: (H, W) uint8 {0, 255}
        prob_map: (H, W) float32 [0, 1]
        save_path: 保存路径（可选）
    Returns:
        vis: (H, W*3, 3) uint8 可视化图
    """
    h, w = image.shape[:2]

    # 二值 mask → RGB
    mask_rgb = np.stack([mask_bin] * 3, axis=-1)

    # 叠加（绿色半透明）
    overlay = image.copy()
    overlay[mask_bin > 0] = (0, 255, 0)  # 绿色

    # 三栏并排
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = image
    vis[:, w:2*w] = mask_rgb
    vis[:, 2*w:3*w] = overlay

    # 标签
    cv2.putText(vis, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(vis, "Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(vis, "Overlay", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    if save_path:
        ensure_dir(save_path.parent)
        if not imwrite_unicode(save_path, vis):
            raise IOError(f"无法保存可视化图像: {save_path}")
        print(f"[Predict] 可视化已保存: {save_path}")

    return vis


def predict_image(image_path, checkpoint_path=None, output_dir=None, tta=False):
    """
    对单张图像进行完整推理（加载 → 预测 → 可视化）

    Args:
        image_path:     图片路径
        checkpoint_path: 模型权重路径
        output_dir:      输出目录（默认为 results/predict）
        tta:            是否启用 TTA
    Returns:
        mask_bin, prob_map, vis 图
    """
    from pathlib import Path
    image_path = Path(image_path)

    if output_dir is None:
        from config import RESULT_DIR
        output_dir = RESULT_DIR / "predict"

    # 加载图像
    image = imread_unicode(image_path)
    if image is None:
        raise IOError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载模型
    model, device = load_model_for_inference(checkpoint_path)

    # 预处理变换
    transform = get_predict_transform()

    # 推理
    prob_map, mask_bin = predict_single(model, device, image, transform, tta=tta)

    # 可视化
    save_path = output_dir / f"{image_path.stem}_result.jpg"
    vis = visualize_prediction(image, mask_bin, prob_map, save_path=save_path)

    # 保存 mask
    mask_save_path = output_dir / f"{image_path.stem}_mask.png"
    if not imwrite_unicode(mask_save_path, mask_bin):
        raise IOError(f"无法保存 Mask: {mask_save_path}")

    print(f"[Predict] 推理完成!")
    print(f"  - 输入图像: {image_path}")
    print(f"  - 图像尺寸: {image.shape[1]}x{image.shape[0]}")
    print(f"  - 分割结果: {save_path}")
    print(f"  - 二值 Mask: {mask_save_path}")

    return mask_bin, prob_map, vis
