import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import time

from data.dataset import SeismicSegmentationDataset
from models.TFFNet import TFFNet
from utils_base.utils import calculate_pixel_error_metrics_gpu


# =========================
# Load model
# =========================
def load_model(model_path, device):
    model = TFFNet(in_channels=1, out_channels=1).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove unnecessary keys if needed
    processed_state_dict = {
        k: v for k, v in state_dict.items()
        if 'weight_dict' not in k
    }

    model.load_state_dict(processed_state_dict, strict=False)
    model.eval()

    return model


# =========================
# Prediction + evaluation
# =========================
def predict_and_evaluate_gpu(model, test_loader, device):
    start_time = time.time()

    model.eval()
    all_metrics = []

    with torch.no_grad():
        for traces, labels in test_loader:
            traces = traces.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(traces)              # [B, 1, H, W]
            outputs = outputs.squeeze(1)         # [B, H, W]

            # Convert to one-hot prediction
            pred_indices = torch.argmax(outputs, dim=1)   # [B, W]
            predicted = torch.zeros_like(outputs)
            predicted.scatter_(1, pred_indices.unsqueeze(1), 1.0)

            metrics = calculate_pixel_error_metrics_gpu(predicted, labels)
            all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0]
    }

    print("\nAverage metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    total_time = time.time() - start_time
    print(f"\nEvaluation time: {total_time:.2f}s")

    return avg_metrics


# =========================
# Main
# =========================
if __name__ == '__main__':
    input_height = 128
    input_width = 256

    test_data_dir = Path('/home/user/data/xsc/hardpicks-main/HDF5/Lalor/test')

    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")

    # Dataset
    test_dataset = SeismicSegmentationDataset(test_data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Samples: {len(test_dataset)}")
    print(f"Batches: {len(test_loader)}")

    # Model directory
    model_dir = Path(
        '/home/user/data/xsc/Unet/TFFNet/results/train_Lalor'
    )

    model_files = sorted(
        model_dir.glob('epoch_*.pth'),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    print(f"Found {len(model_files)} models")

    best_1px = 0.0
    best_epoch = 0
    best_metrics = {}

    for model_path in model_files:
        print(f"\nEvaluating: {model_path.name}")

        model = load_model(str(model_path), device)
        epoch_num = int(model_path.stem.split('_')[-1])

        metrics = predict_and_evaluate_gpu(model, test_loader, device)

        if metrics['within_1px'] > best_1px:
            best_1px = metrics['within_1px']
            best_epoch = epoch_num
            best_metrics = metrics.copy()

        if device == 'cuda':
            torch.cuda.empty_cache()
            del model

    print("\n" + "=" * 50)
    print(f"Best Epoch: {best_epoch}")
    print("=" * 50)

    for k, v in best_metrics.items():
        print(f"{k}: {v:.4f}")