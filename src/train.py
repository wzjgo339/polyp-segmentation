"""
训练引擎 — 包含 train_one_epoch、validate、完整训练流程

功能：
- AMP 自动混合精度训练
- AdamW 优化器（差分学习率）
- CosineAnnealingLR 学习率调度
- 早停机制（patience=20）
- TensorBoard 日志记录
- Best checkpoint 自动保存
- 梯度裁剪
- Encoder 冻结策略（前 N 个 epoch）
"""
import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

import random as _random
from config import (
    NUM_EPOCHS, BATCH_SIZE,
    WEIGHT_DECAY, BETAS, GRAD_CLIP_NORM,
    EARLY_STOP_PATIENCE, FREEZE_ENCODER_EPOCHS,
    USE_AMP, LOSS, EVAL_THRESHOLD,
    CHECKPOINT_DIR, LOG_DIR,
    LR_ENCODER, LR_DECODER,
    SEED, TRAIN_MULTISCALE,
)
from src.model import create_model, get_optimizer_params
from src.loss import get_loss_fn
from src.metrics import dice_coeff, iou_score
from src.utils import set_seed, ensure_dir


def freeze_encoder(model, freeze=True):
    """
    冻结或解冻 encoder 参数
    """
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = not freeze

    status = "冻结" if freeze else "解冻"
    print(f"[Train] Encoder 参数已{status}")


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch, writer, grad_clip_norm=GRAD_CLIP_NORM):
    """
    训练一个 epoch

    Returns:
        avg_loss (float), avg_dice (float)
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if USE_AMP:
            with autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()

            # 梯度裁剪
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        # 统计
        batch_dice = dice_coeff(logits.detach(), masks, threshold=EVAL_THRESHOLD)
        total_loss += loss.item()
        total_dice += batch_dice

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{batch_dice:.4f}",
        })

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/dice", avg_dice, epoch)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer):
    """
    验证一个 epoch

    Returns:
        avg_loss (float), avg_dice (float), avg_iou (float)
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_dice = dice_coeff(logits, masks, threshold=EVAL_THRESHOLD)
        batch_iou = iou_score(logits, masks, threshold=EVAL_THRESHOLD)

        total_loss += loss.item()
        total_dice += batch_dice
        total_iou += batch_iou

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{batch_dice:.4f}",
            "iou": f"{batch_iou:.4f}",
        })

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/dice", avg_dice, epoch)
    writer.add_scalar("val/iou", avg_iou, epoch)

    return avg_loss, avg_dice, avg_iou


def run_training(train_loader, val_loader, train_dataset=None, resume=False):
    """
    完整训练流程

    Args:
        train_loader: 训练 DataLoader
        val_loader:   验证 DataLoader
        train_dataset: 训练集 Dataset（用于 multi-scale 切换）

    Returns:
        model (训练好的模型)
        best_epoch (int): 最佳验证 Dice 对应的 epoch
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(LOG_DIR)

    print(f"\n{'='*50}")
    print(f"[Train] 设备: {device}")
    print(f"[Train] 训练样本: {len(train_loader.dataset)}")
    print(f"[Train] 验证样本: {len(val_loader.dataset)}")
    print(f"[Train] Batch size: {BATCH_SIZE}, AMP: {USE_AMP}")
    print(f"[Train] 最大 epochs: {NUM_EPOCHS}, 早停 patience: {EARLY_STOP_PATIENCE}")
    print(f"{'='*50}\n")

    # 模型
    model = create_model()
    model = model.to(device)

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    checkpoint = None
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"无法续训，未找到 {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        freeze_encoder(model, freeze=False)
        print(f"[Resume] 已加载 checkpoint: epoch {checkpoint['epoch']}")
    else:
        # 初始冻结 encoder
        freeze_encoder(model, freeze=True)

    # 损失函数
    criterion = get_loss_fn(LOSS)

    # 优化器（差分学习率）
    params_group = get_optimizer_params(model)
    optimizer = AdamW(
        params_group,
        lr=LR_DECODER,  # 默认 LR（被分组 LR 覆盖）
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
    )

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 学习率调度
    remaining_epochs = max(1, NUM_EPOCHS - checkpoint["epoch"]) \
        if checkpoint is not None else NUM_EPOCHS
    scheduler = CosineAnnealingLR(
        optimizer, T_max=remaining_epochs, eta_min=1e-6
    )
    if checkpoint is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # AMP GradScaler
    scaler = GradScaler('cuda', enabled=USE_AMP)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    writer.add_text("config", f"batch_size={BATCH_SIZE}, "
                             f"lr_enc={LR_ENCODER}, "
                             f"lr_dec={LR_DECODER}, "
                             f"loss={LOSS}, "
                             f"amp={USE_AMP}")

    # 训练循环
    best_dice = checkpoint.get("best_dice", 0.0) if checkpoint else 0.0
    best_epoch = checkpoint.get("epoch", 0) if checkpoint else 0
    patience_counter = 0
    training_start = time.time()
    start_epoch = checkpoint["epoch"] + 1 if checkpoint else 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Multi-scale: 随机切换训练尺寸
        if TRAIN_MULTISCALE and train_dataset is not None \
                and hasattr(train_dataset, 'ms_transforms'):
            size = _random.choice(list(train_dataset.ms_transforms.keys()))
            train_dataset.transform = train_dataset.ms_transforms[size]

        # Encoder 解冻逻辑
        if epoch == FREEZE_ENCODER_EPOCHS + 1:
            freeze_encoder(model, freeze=False)
            # 解冻后重新设置优化器参数（学习率切换）
            # 注意：优化器已有 encoder 参数，解冻后 param 的 requires_grad 变 True
            # 但优化器已经创建，需重建以包含新参数
            print("[Train] 重建优化器以包含 encoder 参数")
            params_group = get_optimizer_params(model)
            optimizer = AdamW(
                params_group,
                lr=LR_DECODER,
                weight_decay=WEIGHT_DECAY,
                betas=BETAS,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - epoch + 1,
                                          eta_min=1e-6)

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, writer
        )

        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # LR scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch)

        epoch_time = time.time() - epoch_start

        # 打印汇总
        print(f"\n[Epoch {epoch:3d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型（根据 val Dice）
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "best_iou": val_iou,
            }, str(checkpoint_path))
            print(f"[Checkpoint] 保存最佳模型: {checkpoint_path} "
                  f"(Dice={best_dice:.4f}, IoU={val_iou:.4f})")
        else:
            patience_counter += 1
            print(f"[EarlyStop] {patience_counter}/{EARLY_STOP_PATIENCE}")

        # 早停
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[EarlyStop] 已触发早停 (epoch {epoch})")
            break

        print()

    total_time = time.time() - training_start
    print(f"\n{'='*50}")
    print(f"[Train] 训练完成!")
    print(f"[Train] 最佳 epoch: {best_epoch}, 最佳 val Dice: {best_dice:.4f}")
    print(f"[Train] 总训练时间: {total_time / 60:.1f} 分钟")
    print(f"{'='*50}")

    writer.close()

    # 加载最佳模型返回
    best_path = CHECKPOINT_DIR / "best_model.pth"
    if best_path.exists():
        checkpoint = torch.load(str(best_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Train] 已加载最佳模型 (epoch {best_epoch})")

    return model, best_epoch
