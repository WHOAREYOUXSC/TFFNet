import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time


from data.dataset import SeismicSegmentationDataset
from models.TFFNet import TFFNet
from utils_base.utils import set_seed


# =========================
# Configuration
# =========================
class Config:
    seed = 42
    device = 'cuda'

    train_data_dir = '/home/user/data/xsc/hardpicks-main/HDF5/Lalor/train'
    val_data_dir   = '/home/user/data/xsc/hardpicks-main/HDF5/Lalor/valid'

    batch_size = 8
    num_workers = 4
    num_epochs = 50

    # ===== Fixed training settings =====
    optimizer_name = 'Adam'
    lr = 0.001
    weight_decay = 1e-6

    # ===== Extendable optimizer configurations =====
    optimizers = {
        'Adam': {
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'AdamW': {
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'SGD': {
            'momentum': 0.9,
            'nesterov': True
        }
    }

    save_dir = 'results/train_Lalor'


# =========================
# Build optimizer
# =========================
def build_optimizer(model, cfg):
    opt_name = cfg.optimizer_name
    opt_cfg = cfg.optimizers[opt_name]

    if opt_name == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=opt_cfg['betas'],
            eps=opt_cfg['eps']
        )

    elif opt_name == 'AdamW':
        return optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=opt_cfg['betas'],
            eps=opt_cfg['eps']
        )

    elif opt_name == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=opt_cfg['momentum'],
            nesterov=opt_cfg['nesterov']
        )

    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


# =========================
# Training function
# =========================
def train_one_model(model, train_loader, val_loader,
                    criterion, optimizer, scheduler,
                    num_epochs, device, save_dir):

    best_val_loss = float('inf')
    best_model_path = save_dir / "best_model.pth"

    for epoch in range(num_epochs):
        # ===== Training =====
        model.train()
        train_loss = 0.0

        for traces, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            traces, labels = traces.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for traces, labels in val_loader:
                traces, labels = traces.to(device), labels.to(device)
                labels = labels.unsqueeze(1).float()

                outputs = model(traces)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # =========================================
        # Save EVERY epoch
        # =========================================
        epoch_model_path = save_dir / f"epoch_{epoch+1}.pth"

        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'val_loss': val_loss,
        # }, epoch_model_path)
        torch.save(model.state_dict(), epoch_model_path)  # 修改：只保存模型权重


        # =========================================
        # Save BEST model
        # =========================================
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # torch.save({
            #     'epoch': epoch + 1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': val_loss,
            # }, best_model_path)
            torch.save(model.state_dict(), epoch_model_path)

            print(f"  ✓ Best model updated (Val Loss={val_loss:.6f})")

    print(f"\nTraining finished. Best Val Loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")

    return best_model_path


# =========================
# Main function
# =========================
def main():
    cfg = Config()

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)

    print(f"Using device: {device}")
    print(f"Optimizer: {cfg.optimizer_name}, LR={cfg.lr}, WD={cfg.weight_decay}")

    # ===== Dataset loading =====
    train_dataset = SeismicSegmentationDataset(Path(cfg.train_data_dir))
    val_dataset   = SeismicSegmentationDataset(Path(cfg.val_data_dir))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.num_workers
    )

    # ===== Model =====
    model = TFFNet(in_channels=1, out_channels=1).to(device)

    # ===== Loss function =====
    criterion = nn.BCELoss()

    # ===== Optimizer =====
    optimizer = build_optimizer(model, cfg)

    # ===== Learning rate scheduler =====
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs * 2
    )

    # ===== Output directory =====
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== Start training =====
    start_time = time.time()

    train_one_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        cfg.num_epochs, device, save_dir
    )

    print(f"\nTotal time: {(time.time() - start_time)/60:.2f} min")


if __name__ == '__main__':
    main()