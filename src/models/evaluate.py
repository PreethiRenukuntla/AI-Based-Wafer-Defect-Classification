import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from src.config import (
    BEST_MODEL_PATH, BASELINE_ACCURACY, DEFECT_CLASSES, NUM_CLASSES,
    BATCH_SIZE, DEVICE, FIGURES_DIR, REPORTS_DIR, MODELS_DIR
)
from src.data.dataset import get_data_splits
from src.models.architecture import WaferDefectResNet

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def evaluate_official_test_set():
    print("=" * 60)
    print("OFFICIAL TEST SET EVALUATION")
    print("=" * 60)

    # 1. Load Data Splits & Model
    raw_df, meta_df, train_dataset, val_dataset, test_dataset = get_data_splits()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at {BEST_MODEL_PATH}")

    print(f"Loading trained checkpoint from {BEST_MODEL_PATH}...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model = WaferDefectResNet(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []

    print(f"Evaluating model on official test set ({len(test_dataset):,} samples)...")
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

    # 2. Compute Metrics
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
    print("FINAL OFFICIAL TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline Accuracy:        {BASELINE_ACCURACY:.2f}%")
    print(f"New Official Test Acc:   {test_acc:.2f}%")
    print(f"Accuracy Improvement:     +{accuracy_improvement:.2f} percentage points")
    print(f"Error Rate:               {error_pct:.2f}%")
    print(f"Balanced Accuracy:        {balanced_acc:.2f}%")
    print(f"Macro F1 Score:           {f1_macro:.4f}")
    print(f"Weighted F1 Score:        {f1_weighted:.4f}")
    print("=" * 60)

    # 3. Save Classification Report CSV & Model Report JSON
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

    print(f"Reports saved to {REPORTS_DIR}")

    # 4. Generate Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
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

    # Training & Validation Loss/Accuracy Curves
    history_path = MODELS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)

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

    print("Figures saved to outputs/figures/ (confusion_matrix.png, training_curves.png)")
    return model_report

if __name__ == "__main__":
    evaluate_official_test_set()
