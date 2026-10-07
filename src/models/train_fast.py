import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from src.config import (
    BEST_MODEL_PATH, BASELINE_ACCURACY, DEFECT_CLASSES, NUM_CLASSES,
    LEARNING_RATE, WEIGHT_DECAY, SEED, DEVICE,
    FIGURES_DIR, REPORTS_DIR, MODELS_DIR
)
from src.data.fast_dataset import cache_preprocessed_datasets
from src.models.architecture import WaferDefectResNet

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

FAST_BATCH_SIZE = 256
FAST_EPOCHS = 10


class TensorDatasetFast(torch.utils.data.Dataset):
    def __init__(self, images_uint8, labels_int):
        self.images = torch.from_numpy(images_uint8).unsqueeze(1).float() / 255.0
        self.labels = torch.from_numpy(labels_int).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class FocalLoss(nn.Module):
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
        return focal_loss.sum()


def apply_batch_augmentations(images):
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[3])
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[2])
    return images


def run_fast_pipeline():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("RUNNING ULTRA-FAST CPU TRAINING & SACRED TEST SET EVALUATION")
    print("=" * 60)

    train_data, test_data = cache_preprocessed_datasets()

    tr_imgs = train_data['images']
    tr_lbls = train_data['labels']
    te_imgs = test_data['images']
    te_lbls = test_data['labels']

    train_indices, val_indices = train_test_split(
        np.arange(len(tr_lbls)),
        test_size=0.15,
        random_state=SEED,
        stratify=tr_lbls
    )

    train_ds = TensorDatasetFast(tr_imgs[train_indices], tr_lbls[train_indices])
    val_ds = TensorDatasetFast(tr_imgs[val_indices], tr_lbls[val_indices])
    test_ds = TensorDatasetFast(te_imgs, te_lbls)

    print(f"Dataset Split Sizes:")
    print(f"  Train Set: {len(train_ds):,} samples")
    print(f"  Val Set:   {len(val_ds):,} samples")
    print(f"  Test Set:  {len(test_ds):,} samples (Official Sacred Test Set)")

    # Inverse Class Frequency Weights for Focal Loss
    class_counts = np.bincount(tr_lbls[train_indices], minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-5)
    class_weights = class_weights / np.sum(class_weights) * NUM_CLASSES
    weights_tensor = torch.from_numpy(class_weights).float().to(DEVICE)
    weights_tensor = torch.clamp(weights_tensor, min=0.1, max=15.0)

    train_loader = DataLoader(train_ds, batch_size=FAST_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=FAST_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=FAST_BATCH_SIZE, shuffle=False, num_workers=0)

    model = WaferDefectResNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FAST_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    best_val_loss = float('inf')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(1, FAST_EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images = apply_batch_augmentations(images)
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

        print(f"Epoch [{epoch:02d}/{FAST_EPOCHS:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

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
            print(f"  >>> Best Checkpoint Saved! Val Acc: {best_val_acc:.2f}%")

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds. Best Val Acc: {best_val_acc:.2f}%")

    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # 2. EVALUATION ON SACRED OFFICIAL TEST SET (118,595 samples)
    print("\nLoading best model checkpoint for Sacred Official Test Set Evaluation...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    test_acc = accuracy_score(y_true, y_pred) * 100.0
    error_pct = 100.0 - test_acc
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100.0

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, cls_name in enumerate(DEFECT_CLASSES):
        per_class_metrics[cls_name] = {
            'precision': round(float(p_per_class[i]), 4),
            'recall': round(float(r_per_class[i]), 4),
            'f1_score': round(float(f1_per_class[i]), 4),
            'support': int(support_per_class[i])
        }

    accuracy_improvement = test_acc - BASELINE_ACCURACY

    print("\n" + "=" * 60)
    print("SACRED OFFICIAL TEST SET PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Original Baseline Test Acc: {BASELINE_ACCURACY:.2f}%")
    print(f"NEW OFFICIAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"ACCURACY IMPROVEMENT:       +{accuracy_improvement:.2f} percentage points")
    print(f"ERROR PERCENTAGE:            {error_pct:.2f}%")
    print(f"BALANCED ACCURACY:           {balanced_acc:.2f}%")
    print(f"MACRO F1 SCORE:              {f1_macro:.4f}")
    print(f"WEIGHTED F1 SCORE:           {f1_weighted:.4f}")
    print("=" * 60)

    # Save Reports & Figures
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(
        y_true, y_pred, target_names=DEFECT_CLASSES, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORTS_DIR / "classification_report.csv")

    model_report = {
        'baseline_accuracy_pct': BASELINE_ACCURACY,
        'official_test_accuracy_pct': round(float(test_acc), 2),
        'accuracy_improvement_pts': round(float(accuracy_improvement), 2),
        'error_percentage_pct': round(float(error_pct), 2),
        'balanced_accuracy_pct': round(float(balanced_acc), 2),
        'macro_precision': round(float(precision_macro), 4),
        'macro_recall': round(float(recall_macro), 4),
        'macro_f1_score': round(float(f1_macro), 4),
        'weighted_f1_score': round(float(f1_weighted), 4),
        'total_test_samples': int(len(y_true)),
        'per_class_performance': per_class_metrics
    }
    with open(REPORTS_DIR / "model_report.json", "w") as f:
        json.dump(model_report, f, indent=4)

    project_report = {
        'project_title': 'AI-Based Semiconductor Wafer Defect Inspection & Support System',
        'dataset': 'WM-811K (LSWMD.pkl)',
        'baseline_accuracy': f"{BASELINE_ACCURACY:.2f}%",
        'achieved_test_accuracy': f"{test_acc:.2f}%",
        'improvement': f"+{accuracy_improvement:.2f}%",
        'balanced_accuracy': f"{balanced_acc:.2f}%",
        'error_rate': f"{error_pct:.2f}%",
        'per_class_summary': per_class_metrics
    }
    with open(REPORTS_DIR / "project_report.json", "w") as f:
        json.dump(project_report, f, indent=4)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=DEFECT_CLASSES, yticklabels=DEFECT_CLASSES, ax=ax)
    ax.set_title(f"Normalized Confusion Matrix (Official Test Set Acc: {test_acc:.2f}%)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_range = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss', color='crimson')
    ax1.plot(epochs_range, history['val_loss'], 's-', label='Val Loss', color='navy')
    ax1.set_title("Training vs Validation Loss", fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs_range, history['train_acc'], 'o-', label='Train Acc', color='crimson')
    ax2.plot(epochs_range, history['val_acc'], 's-', label='Val Acc', color='navy')
    ax2.axhline(BASELINE_ACCURACY, color='gray', linestyle='--', label=f'Baseline ({BASELINE_ACCURACY}%)')
    ax2.set_title("Training vs Validation Accuracy (%)", fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=300)
    plt.close()

    print("All reports and figures saved successfully!")
    return model_report

if __name__ == "__main__":
    run_fast_pipeline()
