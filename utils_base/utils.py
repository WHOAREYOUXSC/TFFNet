import random
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_loss(train_loss, val_loss, info):
    """Plot training and validation loss."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {info}')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


def set_seed(seed):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_pixel_error_metrics_gpu(pred, target):
    """
    Compute pixel-level picking metrics on GPU.

    pred: [B, 2, H, W] (softmax) or [B, 1, H, W]
    target: [B, 1, H, W] or [B, H, W]
    """

    device = pred.device

    # ===== Prediction processing =====
    if pred.shape[1] == 2:
        # Use argmax to get predicted picking index
        pred_binary = torch.argmax(pred, dim=1)  # [B, H, W] -> class index
    else:
        pred_binary = pred.squeeze(1)

    # Convert to binary mask (only class=1 kept)
    pred_binary = (pred_binary == 1).float()

    # ===== Target processing =====
    if target.dim() == 4:
        target = target.squeeze(1)

    target_binary = (target > 0.5).float()

    B, H, W = pred_binary.shape

    # ===== First arrival picking (along height) =====
    pred_first = torch.argmax(pred_binary, dim=1)     # [B, W]
    target_first = torch.argmax(target_binary, dim=1) # [B, W]

    # Handle no-pick case
    pred_has = torch.any(pred_binary == 1, dim=1)
    target_has = torch.any(target_binary == 1, dim=1)

    pred_first = torch.where(pred_has, pred_first, torch.full_like(pred_first, H))
    target_first = torch.where(target_has, target_first, torch.full_like(target_first, H))

    # ===== Error =====
    errors = pred_first.float() - target_first.float()
    abs_errors = torch.abs(errors)

    total = B * W

    # ===== Accuracy within tolerance =====
    acc_1 = torch.sum(abs_errors <= 1).float() / total
    acc_3 = torch.sum(abs_errors <= 3).float() / total
    acc_5 = torch.sum(abs_errors <= 5).float() / total
    acc_7 = torch.sum(abs_errors <= 7).float() / total
    acc_9 = torch.sum(abs_errors <= 9).float() / total

    # ===== Error metrics =====
    rmse = torch.sqrt(torch.mean(errors ** 2))
    mbe = torch.mean(errors)
    mae = torch.mean(abs_errors)

    return {
        'within_1px': acc_1.item() * 100,
        'within_3px': acc_3.item() * 100,
        'within_5px': acc_5.item() * 100,
        'within_7px': acc_7.item() * 100,
        'within_9px': acc_9.item() * 100,
        'RMSE': rmse.item(),
        'MBE': mbe.item(),
        'MAE': mae.item()
    }