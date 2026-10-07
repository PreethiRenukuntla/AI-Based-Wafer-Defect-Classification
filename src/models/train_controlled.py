import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    SEED,
    DEFECT_CLASSES,
    NUM_CLASSES,
    DEVICE,
    MODELS_DIR,
)

from src.data.dataset import get_data_splits
from src.models.architecture import WaferDefectResNet


# =========================================================
# CONTROLLED RESNET TRAINING
# =========================================================
#
# Training strategy:
#   - Normal shuffled training
#   - No WeightedRandomSampler
#   - No Focal Loss
#   - Moderate class-weighted CrossEntropyLoss
#   - Same official train/validation split
#   - Official test set remains completely untouched
#
# Goal:
#   Train a reliable ResNet model without spending
#   excessive CPU time on unnecessary epochs.
#
# =========================================================


CONTROLLED_MODEL_PATH = (
    MODELS_DIR / "wafer_resnet_controlled_best.pth"
)

HISTORY_PATH = (
    MODELS_DIR / "controlled_training_history.json"
)


# =========================================================
# MODERATE CLASS WEIGHTS
# =========================================================

def create_moderate_class_weights(train_dataset):
    """
    Create moderate class weights using:

        weight = 1 / sqrt(class_frequency)

    This avoids the extreme weighting that can happen
    with direct inverse-frequency weighting.
    """

    labels = [
        train_dataset.metadata.iloc[i]["class_idx"]
        for i in range(len(train_dataset))
    ]

    class_counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    ).astype(np.float32)

    total = class_counts.sum()

    frequencies = class_counts / total

    # Moderate inverse-frequency weighting
    weights = 1.0 / np.sqrt(
        frequencies + 1e-8
    )

    # Normalize around 1.0
    weights = weights / weights.mean()

    # Prevent extreme weights
    weights = np.clip(
        weights,
        0.5,
        5.0
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_model():

    # -----------------------------------------------------
    # REPRODUCIBILITY
    # -----------------------------------------------------

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    print("=" * 70)
    print("CONTROLLED RESNET TRAINING")
    print("=" * 70)

    print()

    print("Training strategy:")
    print("  - No WeightedRandomSampler")
    print("  - No Focal Loss")
    print("  - Moderate class-weighted CrossEntropyLoss")
    print("  - Official test set remains untouched")
    print("  - CPU-friendly 5 epoch training")

    print()

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # -----------------------------------------------------
    # 1. LOAD OFFICIAL DATA SPLITS
    # -----------------------------------------------------

    (
        raw_df,
        meta_df,
        train_dataset,
        val_dataset,
        test_dataset
    ) = get_data_splits(
        val_ratio=0.15,
        seed=SEED
    )

    print()
    print("Dataset sizes:")
    print(
        f"  Train: {len(train_dataset):,}"
    )
    print(
        f"  Val:   {len(val_dataset):,}"
    )
    print(
        f"  Test:  {len(test_dataset):,}"
    )

    print()
    print(
        "IMPORTANT: Official test set is NOT used during training."
    )

    # -----------------------------------------------------
    # 2. DATA LOADERS
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # -----------------------------------------------------
    # 3. CREATE MODERATE CLASS WEIGHTS
    # -----------------------------------------------------

    class_weights = create_moderate_class_weights(
        train_dataset
    ).to(DEVICE)

    print()
    print("Moderate class weights:")

    for idx, class_name in enumerate(DEFECT_CLASSES):

        print(
            f"  {class_name:12s}: "
            f"{class_weights[idx].item():.3f}"
        )

    # -----------------------------------------------------
    # 4. CREATE MODEL
    # -----------------------------------------------------

    model = WaferDefectResNet(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print()
    print("Model:")
    print("  WaferDefectResNet")
    print(
        f"  Number of classes: {NUM_CLASSES}"
    )

    # -----------------------------------------------------
    # 5. LOSS FUNCTION
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # 6. OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # -----------------------------------------------------
    # 7. LEARNING RATE SCHEDULER
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5
    )

    # -----------------------------------------------------
    # 8. TRACK BEST MODEL
    # -----------------------------------------------------

    best_val_acc = 0.0
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # -----------------------------------------------------
    # 9. START TIMER
    # -----------------------------------------------------

    start_time = time.time()

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # =================================================
        # TRAINING
        # =================================================

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            train_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += images.size(0)

        # -------------------------------------------------
        # Update learning rate
        # -------------------------------------------------

        scheduler.step()

        # -------------------------------------------------
        # Training metrics
        # -------------------------------------------------

        epoch_train_loss = (
            train_loss / train_total
        )

        epoch_train_acc = (
            train_correct
            / train_total
            * 100.0
        )

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += images.size(0)

        # -------------------------------------------------
        # Validation metrics
        # -------------------------------------------------

        epoch_val_loss = (
            val_loss / val_total
        )

        epoch_val_acc = (
            val_correct
            / val_total
            * 100.0
        )

        # =================================================
        # SAVE HISTORY
        # =================================================

        history["train_loss"].append(
            epoch_train_loss
        )

        history["train_acc"].append(
            epoch_train_acc
        )

        history["val_loss"].append(
            epoch_val_loss
        )

        history["val_acc"].append(
            epoch_val_acc
        )

        # =================================================
        # PRINT RESULTS
        # =================================================

        print()

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
            f"| Train Loss: {epoch_train_loss:.4f} "
            f"| Train Acc: {epoch_train_acc:.2f}% "
            f"| Val Loss: {epoch_val_loss:.4f} "
            f"| Val Acc: {epoch_val_acc:.2f}%"
        )

        print(
            f"  Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.7f}"
        )

        # =================================================
        # SAVE BEST CHECKPOINT
        # =================================================

        if (
            epoch_val_acc > best_val_acc
            or (
                epoch_val_acc == best_val_acc
                and epoch_val_loss < best_val_loss
            )
        ):

            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss

            MODELS_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": best_val_loss,
                "classes": DEFECT_CLASSES
            }

            torch.save(
                checkpoint,
                CONTROLLED_MODEL_PATH
            )

            print(
                "  >>> BEST CHECKPOINT SAVED"
            )

            print(
                f"  >>> {CONTROLLED_MODEL_PATH}"
            )

            print(
                f"  >>> Validation Accuracy: "
                f"{best_val_acc:.2f}%"
            )

    # =====================================================
    # TRAINING COMPLETE
    # =====================================================

    total_time = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print("CONTROLLED TRAINING COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print()

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )

    print()

    print(
        "Best model saved to:"
    )

    print(
        CONTROLLED_MODEL_PATH
    )

    # =====================================================
    # SAVE TRAINING HISTORY
    # =====================================================

    with open(
        HISTORY_PATH,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    print()

    print(
        "Training history saved to:"
    )

    print(
        HISTORY_PATH
    )

    print()

    print("=" * 70)
    print("NEXT STEP: TEST-SET EVALUATION")
    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    train_model()