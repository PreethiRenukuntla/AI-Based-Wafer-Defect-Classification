"""
Official test evaluation for the controlled ResNet model.

Model:
    models/wafer_resnet_controlled_best.pth

Evaluation:
    Official WM-811K test set
    118,595 samples

Outputs:
    outputs/controlled_official_test_results.json
    outputs/controlled_official_test_results.txt
    outputs/controlled_confusion_matrix.npy
"""

from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_CLASSES,
    DEFECT_CLASSES,
    OUTPUTS_DIR,
)
from src.data.dataset import get_data_splits
from src.models.architecture import WaferDefectResNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/wafer_resnet_controlled_best.pth")

JSON_OUTPUT = OUTPUTS_DIR / "controlled_official_test_results.json"
TXT_OUTPUT = OUTPUTS_DIR / "controlled_official_test_results.txt"
CM_OUTPUT = OUTPUTS_DIR / "controlled_confusion_matrix.npy"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_test_dataset():
    """
    Get the official test dataset from get_data_splits().

    Different versions of the project may return different
    numbers of split objects. The official test dataset is
    expected to be the final returned object.
    """

    splits = get_data_splits()

    if not isinstance(splits, (tuple, list)):
        raise TypeError(
            "get_data_splits() did not return a tuple/list. "
            f"Returned type: {type(splits)}"
        )

    if len(splits) < 3:
        raise ValueError(
            "get_data_splits() returned fewer than 3 objects. "
            f"Returned {len(splits)} objects."
        )

    test_dataset = splits[-1]

    return test_dataset


def load_checkpoint(model, checkpoint_path):
    """
    Load a PyTorch checkpoint while supporting several common
    checkpoint formats.
    """

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Could already be a raw state_dict
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)

    print("Checkpoint loaded successfully.")

    return model


def get_batch_data(batch):
    """
    Handle common Dataset return formats.

    Expected:
        (images, labels)

    Also supports:
        [images, labels]
    """

    if isinstance(batch, (tuple, list)):

        if len(batch) >= 2:
            images = batch[0]
            labels = batch[1]
            return images, labels

    raise ValueError(
        "Unexpected batch format. "
        "Expected at least (images, labels)."
    )


def safe_class_name(index):
    """
    Return class name safely.
    """

    if index < len(DEFECT_CLASSES):
        return DEFECT_CLASSES[index]

    return f"class_{index}"


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("CONTROLLED RESNET — OFFICIAL TEST EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n{MODEL_PATH}\n\n"
            "Make sure the controlled training completed successfully."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load official test dataset
    # --------------------------------------------------------

    print("\nLoading official dataset splits...")

    test_dataset = extract_test_dataset()

    print("\nOfficial test dataset loaded.")

    print(f"Test samples: {len(test_dataset):,}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {len(test_loader):,}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating ResNet model...")

    model = WaferDefectResNet(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model = load_checkpoint(
        model,
        MODEL_PATH,
    )

    model.eval()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING OFFICIAL TEST EVALUATION")
    print("=" * 70)

    print(
        "\nIMPORTANT: The official test set is being evaluated "
        "without training or validation updates."
    )

    all_predictions = []
    all_targets = []

    total_samples = 0

    start_time = time.time()

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            images, labels = get_batch_data(batch)

            images = images.to(
                DEVICE,
                non_blocking=False,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=False,
            )

            # Forward pass
            outputs = model(images)

            # Predicted class
            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_targets.extend(
                labels.cpu().numpy().tolist()
            )

            total_samples += labels.size(0)

            # Progress every 100 batches
            if (
                batch_idx == 0
                or (batch_idx + 1) % 100 == 0
                or batch_idx == len(test_loader) - 1
            ):

                elapsed = time.time() - start_time

                print(
                    f"Progress: "
                    f"{batch_idx + 1:,}/{len(test_loader):,} batches | "
                    f"{total_samples:,}/{len(test_dataset):,} samples | "
                    f"Elapsed: {elapsed / 60:.1f} min"
                )

    elapsed_time = time.time() - start_time

    # --------------------------------------------------------
    # Convert predictions
    # --------------------------------------------------------

    y_true = np.asarray(
        all_targets,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if len(y_true) != len(test_dataset):

        print(
            "\nWARNING:"
            f" expected {len(test_dataset):,} samples but "
            f"evaluated {len(y_true):,} samples."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    error_rate = 1.0 - accuracy

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    class_names = [
        safe_class_name(i)
        for i in range(NUM_CLASSES)
    ]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    np.save(
        CM_OUTPUT,
        cm,
    )

    # --------------------------------------------------------
    # Per-class prediction distribution
    # --------------------------------------------------------

    true_distribution = {}
    predicted_distribution = {}

    for class_idx in range(NUM_CLASSES):

        class_name = safe_class_name(class_idx)

        true_count = int(
            np.sum(y_true == class_idx)
        )

        predicted_count = int(
            np.sum(y_pred == class_idx)
        )

        true_distribution[class_name] = true_count
        predicted_distribution[class_name] = predicted_count

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OFFICIAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTest samples:        {len(y_true):,}"
    )

    print(
        f"Accuracy:            {accuracy * 100:.2f}%"
    )

    print(
        f"Error rate:          {error_rate * 100:.2f}%"
    )

    print(
        f"Balanced accuracy:   {balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro precision:     {macro_precision:.4f}"
    )

    print(
        f"Macro recall:        {macro_recall:.4f}"
    )

    print(
        f"Macro F1:            {macro_f1:.4f}"
    )

    print(
        f"Weighted precision:  {weighted_precision:.4f}"
    )

    print(
        f"Weighted recall:     {weighted_recall:.4f}"
    )

    print(
        f"Weighted F1:         {weighted_f1:.4f}"
    )

    print(
        f"Evaluation time:     {elapsed_time / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(report_text)

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRUE VS PREDICTED CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        f"{'Class':<15}"
        f"{'True':>12}"
        f"{'Predicted':>15}"
    )

    print("-" * 45)

    for class_name in class_names:

        print(
            f"{class_name:<15}"
            f"{true_distribution[class_name]:>12,}"
            f"{predicted_distribution[class_name]:>15,}"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results = {
        "model": str(MODEL_PATH),
        "device": DEVICE,

        "test_samples": int(len(y_true)),

        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),

        "error_rate": float(error_rate),
        "error_percent": float(error_rate * 100),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "balanced_accuracy_percent": float(
            balanced_accuracy * 100
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "evaluation_time_minutes": float(
            elapsed_time / 60
        ),

        "classes": class_names,

        "classification_report": report_dict,

        "true_distribution": true_distribution,

        "predicted_distribution": predicted_distribution,

        "confusion_matrix": cm.tolist(),
    }

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save TXT report
    # --------------------------------------------------------

    with open(
        TXT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "CONTROLLED RESNET — OFFICIAL TEST EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Model: {MODEL_PATH}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Test samples: {len(y_true):,}\n\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Error rate: {error_rate * 100:.2f}%\n"
        )

        f.write(
            f"Balanced accuracy: "
            f"{balanced_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Macro precision: {macro_precision:.4f}\n"
        )

        f.write(
            f"Macro recall: {macro_recall:.4f}\n"
        )

        f.write(
            f"Macro F1: {macro_f1:.4f}\n"
        )

        f.write(
            f"Weighted precision: "
            f"{weighted_precision:.4f}\n"
        )

        f.write(
            f"Weighted recall: "
            f"{weighted_recall:.4f}\n"
        )

        f.write(
            f"Weighted F1: "
            f"{weighted_f1:.4f}\n"
        )

        f.write(
            f"Evaluation time: "
            f"{elapsed_time / 60:.2f} minutes\n\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "-" * 70
            + "\n\n"
        )

        f.write(
            report_text
        )

        f.write(
            "\n\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "TRUE VS PREDICTED DISTRIBUTION\n"
        )

        f.write(
            "-" * 70
            + "\n\n"
        )

        for class_name in class_names:

            f.write(
                f"{class_name:<15}"
                f" True={true_distribution[class_name]:>8,}"
                f"  Predicted={predicted_distribution[class_name]:>8,}\n"
            )

        f.write(
            "\n\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "CONFUSION MATRIX\n"
        )

        f.write(
            "-" * 70
            + "\n\n"
        )

        f.write(
            np.array2string(
                cm,
                separator=" ",
            )
        )

        f.write(
            "\n"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print("\nSaved files:")

    print(
        f"  1. {JSON_OUTPUT}"
    )

    print(
        f"  2. {TXT_OUTPUT}"
    )

    print(
        f"  3. {CM_OUTPUT}"
    )

    print("\nFinal official test accuracy:")
    print(
        f"  >>> {accuracy * 100:.2f}% <<<"
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()