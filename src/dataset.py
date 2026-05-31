"""
数据集定义和数据加载管线

注意：
- TrainDataset 中的图片目录实际名为 "image"（单数）而非 "images"（复数）
- Mask 像素值为 {0, 255}，加载时归一化为 {0.0, 1.0}
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from config import (
    TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TEST_SUBSETS,
    INPUT_HEIGHT, INPUT_WIDTH, TRAIN_VAL_SPLIT,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    AUGMENTATION, NORMALIZE_MEAN, NORMALIZE_STD,
    SEED, TRAIN_MULTISCALE, MULTISCALE_SIZES,
)
from src.utils import normalize_mask, get_image_files


class PolypDataset(Dataset):
    """
    息肉分割数据集

    支持两种模式：
    - train: 同时加载 image + mask，带数据增强
    - test: 加载 image + mask，仅做 resize + 归一化（不增强）
    - predict: 仅加载 image，不加载 mask
    """

    def __init__(self, image_dir, mask_dir=None, mode="train",
                 transform=None, image_files=None):
        """
        Args:
            image_dir:  图片目录路径
            mask_dir:   mask 目录路径（predict 模式下为 None）
            mode:       "train" | "val" | "test" | "predict"
            transform:  albumentations 变换管线
            image_files: 指定文件列表（None = 扫描全目录）
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mode = mode
        self.transform = transform

        self.image_files = get_image_files(self.image_dir)

        # 如果传入了 image_files, 过滤到只在白名单中的文件
        if image_files is not None:
            allowed = {Path(f).name for f in image_files}
            self.image_files = sorted(
                f for f in self.image_files if Path(f).name in allowed
            )

        # 验证 mask 文件完整性
        if mask_dir is not None:
            mask_files = get_image_files(self.mask_dir)
            # 根据共同 basename 匹配
            image_names = {Path(f).stem for f in self.image_files}
            mask_names = {Path(f).stem for f in mask_files}
            common = image_names & mask_names
            assert len(common) > 0, (
                f"image 和 mask 目录没有共同文件！"
                f"示例 images: {list(image_names)[:3]}, "
                f"示例 masks: {list(mask_names)[:3]}"
            )
            # 只保留共同文件
            self.image_files = sorted(
                [f for f in self.image_files if Path(f).stem in common]
            )

        if len(self.image_files) == 0:
            raise FileNotFoundError(
                f"在 {image_dir} 未找到任何图片文件"
            )

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
        """
        使用 np.fromfile + cv2.imdecode 读取图像，
        解决 Windows 中文路径编码问题。
        """
        path_str = str(path)
        stream = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(stream, flags)
        if img is None:
            raise IOError(f"无法读取图片: {path_str}")
        return img

    def __getitem__(self, idx):
        # 加载图片
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        image = self._imread_unicode(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB

        result = {"image": image}

        # 加载 mask（非 predict 模式）
        if self.mask_dir is not None:
            mask_path = self.mask_dir / img_name
            mask = self._imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            # {0, 255} → {0.0, 1.0}
            mask = normalize_mask(mask)  # (H, W) float32
            result["mask"] = mask

        # 应用变换
        if self.transform is not None:
            augmented = self.transform(image=result["image"],
                                       mask=result.get("mask"))
            result["image"] = augmented["image"]
            if "mask" in augmented:
                result["mask"] = augmented["mask"]

        # 确保有 mask 时也加上额外维度
        if "mask" in result and result["mask"].ndim == 2:
            result["mask"] = result["mask"].unsqueeze(0)  # (1, H, W)

        result["name"] = img_name  # 保存文件名用于调试

        return result


def get_train_transform(target_size=None):
    """训练数据增强管线"""
    h, w = target_size or (INPUT_HEIGHT, INPUT_WIDTH)
    aug = AUGMENTATION
    return A.Compose([
        A.Resize(height=h, width=w),
        A.RandomRotate90(p=aug["random_rotate90_prob"]),
        A.HorizontalFlip(p=aug["horizontal_flip_prob"]),
        A.ShiftScaleRotate(
            shift_limit=aug["shift_limit"],
            scale_limit=aug["scale_limit"],
            rotate_limit=aug["rotate_limit"],
            p=aug["shift_scale_rotate_prob"],
        ),
        A.ElasticTransform(
            alpha=aug["elastic_alpha"],
            sigma=aug["elastic_sigma"],
            alpha_affine=aug["elastic_alpha_affine"],
            p=aug["elastic_prob"],
        ),
        A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_limit"],
            contrast_limit=aug["contrast_limit"],
            p=aug["brightness_contrast_prob"],
        ),
        A.GaussianBlur(
            blur_limit=aug["gaussian_blur_limit"],
            p=aug["gaussian_blur_prob"],
        ),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def get_val_transform():
    """验证/测试变换管线（仅 resize + 归一化）"""
    return A.Compose([
        A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def get_predict_transform():
    """推理变换管线（仅 resize + 归一化）"""
    return A.Compose([
        A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def create_dataloaders():
    """
    创建训练和验证 DataLoader（无共享状态，支持 multi-scale）

    从 TrainDataset 中按 9:1 随机分割
    返回: (train_loader, val_loader, train_dataset)
    """
    import random as _random
    from pathlib import Path

    # 扫描全部文件
    all_files = sorted(get_image_files(TRAIN_IMAGE_DIR))

    # 过滤出有对应 mask 的文件
    mask_names = {Path(f).stem for f in get_image_files(TRAIN_MASK_DIR)}
    all_files = sorted(
        f for f in all_files if Path(f).stem in mask_names
    )

    # 9:1 随机分割（固定种子保证可复现）
    rng = _random.Random(SEED)
    rng.shuffle(all_files)
    train_size = int(len(all_files) * TRAIN_VAL_SPLIT)
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]

    # 创建独立的 Dataset 实例（无共享 state 问题！）
    train_dataset = PolypDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        mode="train",
        transform=get_train_transform(),
        image_files=train_files,
    )
    val_dataset = PolypDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        mode="val",
        transform=get_val_transform(),
        image_files=val_files,
    )

    # 预构建 multi-scale transforms
    if TRAIN_MULTISCALE:
        train_dataset.ms_transforms = {
            size: get_train_transform(target_size=size)
            for size in MULTISCALE_SIZES
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset


def create_test_loader(subset_name):
    """
    创建单个测试集的 DataLoader

    Args:
        subset_name: "CVC-300" | "CVC-ClinicDB" | "CVC-ColonDB"
                    | "ETIS-LaribPolypDB" | "Kvasir"
    Returns:
        DataLoader
    """
    subset_dir = TEST_SUBSETS[subset_name]
    image_dir = subset_dir / "images"
    mask_dir = subset_dir / "masks"

    dataset = PolypDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        mode="test",
        transform=get_val_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    return loader


def create_all_test_loaders():
    """
    创建全部 5 个测试集的 DataLoader

    Returns:
        dict: {subset_name: DataLoader}
    """
    loaders = {}
    for name in TEST_SUBSETS:
        loaders[name] = create_test_loader(name)
    return loaders
