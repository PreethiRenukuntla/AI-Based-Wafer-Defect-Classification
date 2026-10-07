import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from src.config import (
    BEST_MODEL_PATH, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, SEED, DEFECT_CLASSES, NUM_CLASSES, DEVICE, MODELS_DIR
)
from src.data.dataset import get_data_splits, create_weighted_sampler
from src.models.architecture import WaferDefectResNet


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance in wafer maps."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("STARTING CPU-OPTIMIZED MODEL TRAINING")
    print("=" * 60)

    # 1. Load Data Splits
    raw_df, meta_df, train_dataset, val_dataset, test_dataset = get_data_splits(val_ratio=0.15, seed=SEED)

    # 2. Weighted Sampler & Dataloaders
    sampler, class_weights = create_weighted_sampler(train_dataset)
    weights_tensor = torch.from_numpy(class_weights).float().to(DEVICE)
    # Clamp extreme weights for stability
    weights_tensor = torch.clamp(weights_tensor, min=0.1, max=20.0)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Model, Criterion, Optimizer, Scheduler
    model = WaferDefectResNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Save Best Checkpoint
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': best_val_loss,
                'classes': DEFECT_CLASSES
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"  >>> Checkpoint Saved to {BEST_MODEL_PATH} (Val Acc: {best_val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time / 60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Save training history
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    return model, history


if __name__ == "__main__":
    train_model()
