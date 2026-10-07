"""
Standalone training + evaluation script.
Uses pre-built cache_train.npz / cache_test.npz.
No DataLoader workers, no LSWMD.pkl I/O during training.
"""
import sys, os, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
from src.config import (
    BEST_MODEL_PATH, BASELINE_ACCURACY, DEFECT_CLASSES, NUM_CLASSES,
    LEARNING_RATE, WEIGHT_DECAY, SEED, DEVICE,
    FIGURES_DIR, REPORTS_DIR, MODELS_DIR,
    DATA_PROCESSED_DIR,
)
from src.models.architecture import WaferDefectResNet

# ── hyper-parameters ───────────────────────────────────────────────────────────
BATCH  = 256
EPOCHS = 12

TRAIN_NPZ = DATA_PROCESSED_DIR / "cache_train.npz"
TEST_NPZ  = DATA_PROCESSED_DIR / "cache_test.npz"

# ── Dataset ────────────────────────────────────────────────────────────────────
class CacheDS(Dataset):
    def __init__(self, imgs_u8, labels, augment=False):
        # imgs_u8: uint8 (N,H,W)  →  float (N,1,H,W) in [0,1]
        self.x = torch.from_numpy(imgs_u8).unsqueeze(1).float().div_(255.0)
        self.y = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        if self.augment:
            if torch.rand(1) > .5: x = x.flip(2)        # vertical
            if torch.rand(1) > .5: x = x.flip(1)        # horizontal  (dims off-by-one: dim 1=H, 2=W for 3-D tensor C,H,W)
            k = int(torch.randint(0, 4, (1,)))
            if k: x = torch.rot90(x, k, [1, 2])
        return x, y

# ── Focal Loss ──────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ── helpers ────────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, acc):
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=DEFECT_CLASSES, yticklabels=DEFECT_CLASSES, ax=ax)
    ax.set_title(f"Normalised Confusion Matrix  (Test Acc: {acc:.2f}%)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

def save_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history['train_loss']) + 1)
    ax1.plot(ep, history['train_loss'], 'o-', color='crimson', label='Train')
    ax1.plot(ep, history['val_loss'],   's-', color='navy',    label='Val')
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(ep, history['train_acc'],  'o-', color='crimson', label='Train')
    ax2.plot(ep, history['val_acc'],    's-', color='navy',    label='Val')
    ax2.axhline(BASELINE_ACCURACY, color='gray', ls='--',
                label=f'Baseline {BASELINE_ACCURACY}%')
    ax2.set_title("Accuracy (%)"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close()

# ── main ────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("=" * 62)
    print("  LOADING PRE-BUILT CACHE ARRAYS")
    print("=" * 62)
    tr_data = np.load(TRAIN_NPZ)
    te_data = np.load(TEST_NPZ)

    tr_imgs  = tr_data['images']   # uint8 (N,56,56)
    tr_lbls  = tr_data['labels']   # int64 (N,)
    te_imgs  = te_data['images']
    te_lbls  = te_data['labels']

    print(f"Train cache: {tr_imgs.shape}, labels={tr_lbls.shape}")
    print(f"Test  cache: {te_imgs.shape}, labels={te_lbls.shape}")

    # stratified val split
    tr_idx, val_idx = train_test_split(
        np.arange(len(tr_lbls)), test_size=0.15,
        random_state=SEED, stratify=tr_lbls)

    train_ds = CacheDS(tr_imgs[tr_idx],  tr_lbls[tr_idx],  augment=True)
    val_ds   = CacheDS(tr_imgs[val_idx], tr_lbls[val_idx], augment=False)
    test_ds  = CacheDS(te_imgs,          te_lbls,           augment=False)

    print(f"\nSplit  Train={len(train_ds):,}  Val={len(val_ds):,}  "
          f"Test={len(test_ds):,} (sacred official test set)")

    # class weights → Focal Loss
    cnt = np.bincount(tr_lbls[tr_idx], minlength=NUM_CLASSES).astype(float)
    w   = 1.0 / (cnt + 1e-5)
    w   = np.clip(w / w.sum() * NUM_CLASSES, 0.1, 15.0)
    w_t = torch.tensor(w, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model     = WaferDefectResNet(num_classes=NUM_CLASSES)
    criterion = FocalLoss(alpha=w_t, gamma=2.0)
    optim     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    history = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc']}

    print("\n" + "=" * 62)
    print("  TRAINING WaferDefectResNet  (CPU)")
    print("=" * 62)
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for x, y in train_loader:
            optim.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            tl += loss.item() * x.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tt += x.size(0)
        sched.step()

        # ── val ──
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out  = model(x)
                loss = criterion(out, y)
                vl += loss.item() * x.size(0)
                vc += (out.argmax(1) == y).sum().item()
                vt += x.size(0)

        ep_tl = tl / tt;  ep_ta = tc / tt * 100
        ep_vl = vl / vt;  ep_va = vc / vt * 100
        for k, v in zip(history, [ep_tl, ep_ta, ep_vl, ep_va]):
            history[k].append(v)

        print(f"Ep {ep:02d}/{EPOCHS}  "
              f"TrainLoss={ep_tl:.4f} TrainAcc={ep_ta:.2f}%  "
              f"ValLoss={ep_vl:.4f} ValAcc={ep_va:.2f}%")

        # ── save best ──
        if ep_va > best_val_acc or (ep_va == best_val_acc and ep_vl < best_val_loss):
            best_val_acc, best_val_loss = ep_va, ep_vl
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': best_val_loss,
                'classes': DEFECT_CLASSES,
            }, BEST_MODEL_PATH)
            print(f"  ✓ checkpoint saved  (val_acc={best_val_acc:.2f}%)")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min   best_val_acc={best_val_acc:.2f}%")

    # save history
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    save_training_curves(history)

    # ── EVALUATION ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  LOADING BEST CHECKPOINT → SACRED OFFICIAL TEST SET")
    print("=" * 62)
    ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    yt, yp, yprob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            out  = model(x)
            prob = F.softmax(out, 1)
            yt.extend(y.numpy()); yp.extend(out.argmax(1).numpy())
            yprob.extend(prob.numpy())

    yt = np.array(yt); yp = np.array(yp); yprob = np.array(yprob)

    test_acc   = accuracy_score(yt, yp) * 100
    err_pct    = 100 - test_acc
    bal_acc    = balanced_accuracy_score(yt, yp) * 100
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average='macro',    zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
    pp, rp, fp, sp = precision_recall_fscore_support(yt, yp, average=None,       zero_division=0)

    per_class = {
        c: {'precision': round(float(pp[i]),4), 'recall': round(float(rp[i]),4),
            'f1_score':  round(float(fp[i]),4), 'support': int(sp[i])}
        for i, c in enumerate(DEFECT_CLASSES)
    }
    improvement = test_acc - BASELINE_ACCURACY

    print("\n" + "=" * 62)
    print("  SACRED OFFICIAL TEST SET  —  FINAL RESULTS")
    print("=" * 62)
    print(f"  Original Baseline Acc   : {BASELINE_ACCURACY:.2f}%")
    print(f"  NEW Test Accuracy       : {test_acc:.2f}%")
    print(f"  Improvement             : +{improvement:.2f} pp")
    print(f"  Error %                 : {err_pct:.2f}%")
    print(f"  Balanced Accuracy       : {bal_acc:.2f}%")
    print(f"  Macro   Precision/Recall/F1 : {pm:.4f} / {rm:.4f} / {fm:.4f}")
    print(f"  Weighted F1             : {fw:.4f}")
    print("  Per-class:")
    for c, m in per_class.items():
        print(f"    {c:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1_score']:.3f}  n={m['support']}")
    print("=" * 62)

    # ── save reports ─────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_report = {
        'baseline_accuracy_pct':     BASELINE_ACCURACY,
        'official_test_accuracy_pct': round(float(test_acc),  2),
        'accuracy_improvement_pts':  round(float(improvement),2),
        'error_percentage_pct':      round(float(err_pct),    2),
        'balanced_accuracy_pct':     round(float(bal_acc),    2),
        'macro_precision':           round(float(pm), 4),
        'macro_recall':              round(float(rm), 4),
        'macro_f1_score':            round(float(fm), 4),
        'weighted_f1_score':         round(float(fw), 4),
        'total_test_samples':        int(len(yt)),
        'per_class_performance':     per_class,
    }
    (REPORTS_DIR / "model_report.json").write_text(json.dumps(model_report, indent=2))

    project_report = {
        'project_title':         'AI-Based Semiconductor Wafer Defect Inspection System',
        'dataset':               'WM-811K (LSWMD.pkl)',
        'baseline_accuracy':     f"{BASELINE_ACCURACY:.2f}%",
        'achieved_test_accuracy':f"{test_acc:.2f}%",
        'improvement':           f"+{improvement:.2f}%",
        'balanced_accuracy':     f"{bal_acc:.2f}%",
        'error_rate':            f"{err_pct:.2f}%",
        'per_class_summary':     per_class,
    }
    (REPORTS_DIR / "project_report.json").write_text(json.dumps(project_report, indent=2))

    # classification report CSV
    rpt_dict = classification_report(yt, yp, target_names=DEFECT_CLASSES,
                                      output_dict=True, zero_division=0)
    pd.DataFrame(rpt_dict).T.to_csv(REPORTS_DIR / "classification_report.csv")

    # insights stub
    insights = {
        'dominant_defect': DEFECT_CLASSES[int(np.bincount(yp[yt!=8]).argmax())] if len(yp[yt!=8]) else 'N/A',
        'note': 'Insights generated from model predictions on the sacred WM-811K official test set.',
    }
    (REPORTS_DIR / "insights.json").write_text(json.dumps(insights, indent=2))

    print(f"\nAll reports saved to {REPORTS_DIR}")

    # confusion matrix figure
    save_confusion_matrix(yt, yp, test_acc)
    print(f"Figures saved to {FIGURES_DIR}")

    return model_report

if __name__ == "__main__":
    main()
