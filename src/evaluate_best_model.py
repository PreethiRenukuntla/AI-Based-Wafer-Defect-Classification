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
    f1_score,
    precision_score,
    recall_score
)

from src.config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    DEVICE,
    DEFECT_CLASSES,
    OUTPUTS_DIR
)
from src.data.dataset import get_data_splits
from src.models.architecture import WaferDefectResNet


def evaluate_resnet():
    print("=" * 70)
    print("OFFICIAL WM-811K RESNET EVALUATION")
    print("=" * 70)

    device = torch.device(DEVICE)

    print(f"Device: {device}")
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"Classes: {len(DEFECT_CLASSES)}")
    print()

    # ---------------------------------------------------------
    # 1. Load the exact project datasets
    # ---------------------------------------------------------
    print("Loading WM-811K dataset and official test split...")

    _, _, _, _, test_dataset = get_data_splits(
        val_ratio=0.15,
        seed=42
    )

    print()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ---------------------------------------------------------
    # 2. Load the trained ResNet checkpoint
    # ---------------------------------------------------------
    print("Loading trained ResNet checkpoint...")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )

    checkpoint_classes = checkpoint.get(
        "classes",
        DEFECT_CLASSES
    )

    if checkpoint_classes != DEFECT_CLASSES:
        raise ValueError(
            "Checkpoint class order does not match config.py!\n"
            f"Checkpoint: {checkpoint_classes}\n"
            f"Config:     {DEFECT_CLASSES}"
        )

    model = WaferDefectResNet(
        num_classes=len(DEFECT_CLASSES),
        in_channels=1
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint.get('val_acc', 'unknown')}"
    )

    print()

    # ---------------------------------------------------------
    # 3. Official test evaluation
    # ---------------------------------------------------------
    print("=" * 70)
    print("EVALUATING OFFICIAL 118,595-SAMPLE TEST SET")
    print("=" * 70)

    all_labels = []
    all_predictions = []

    start_time = time.time()

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(
            test_loader,
            start=1
        ):

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            if batch_idx % 250 == 0:
                processed = min(
                    batch_idx * BATCH_SIZE,
                    len(test_dataset)
                )

                print(
                    f"Processed "
                    f"{processed:,} / "
                    f"{len(test_dataset):,}"
                )

    elapsed = time.time() - start_time

    all_labels = np.asarray(all_labels)
    all_predictions = np.asarray(all_predictions)

    # ---------------------------------------------------------
    # 4. Calculate metrics
    # ---------------------------------------------------------
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    error_percentage = (
        1.0 - accuracy
    ) * 100.0

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    # ---------------------------------------------------------
    # 5. Classification report
    # ---------------------------------------------------------
    report_text = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(DEFECT_CLASSES))),
        target_names=DEFECT_CLASSES,
        zero_division=0
    )

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(DEFECT_CLASSES))),
        target_names=DEFECT_CLASSES,
        zero_division=0,
        output_dict=True
    )

    # ---------------------------------------------------------
    # 6. Confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(DEFECT_CLASSES)))
    )

    # ---------------------------------------------------------
    # 7. Print final results
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("FINAL OFFICIAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test samples:              {len(all_labels):,}"
    )

    print(
        f"Accuracy:                  "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Error percentage:          "
        f"{error_percentage:.2f}%"
    )

    print(
        f"Balanced accuracy:         "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro precision:           "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro recall:              "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1:                  "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted precision:        "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted recall:           "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1:               "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Evaluation time:           "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report_text)

    # ---------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------
    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "model": "WaferDefectResNet",
        "checkpoint": str(BEST_MODEL_PATH),
        "test_samples": int(len(all_labels)),
        "accuracy_percent": float(accuracy * 100),
        "error_percent": float(error_percentage),
        "balanced_accuracy_percent": float(
            balanced_accuracy * 100
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "evaluation_time_minutes": float(
            elapsed / 60
        ),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

    json_path = (
        OUTPUTS_DIR /
        "resnet_official_test_results.json"
    )

    txt_path = (
        OUTPUTS_DIR /
        "resnet_official_test_results.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4
        )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "WM-811K OFFICIAL TEST RESULTS\n"
        )

        f.write(
            "=" * 70 +
            "\n\n"
        )

        f.write(
            f"Model: WaferDefectResNet\n"
        )

        f.write(
            f"Test samples: {len(all_labels):,}\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Error: {error_percentage:.2f}%\n"
        )

        f.write(
            f"Balanced Accuracy: "
            f"{balanced_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Macro Precision: "
            f"{macro_precision:.4f}\n"
        )

        f.write(
            f"Macro Recall: "
            f"{macro_recall:.4f}\n"
        )

        f.write(
            f"Macro F1: "
            f"{macro_f1:.4f}\n"
        )

        f.write(
            f"Weighted Precision: "
            f"{weighted_precision:.4f}\n"
        )

        f.write(
            f"Weighted Recall: "
            f"{weighted_recall:.4f}\n"
        )

        f.write(
            f"Weighted F1: "
            f"{weighted_f1:.4f}\n"
        )

        f.write(
            f"Evaluation Time: "
            f"{elapsed / 60:.2f} minutes\n\n"
        )

        f.write(
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "=" * 70 +
            "\n\n"
        )

        f.write(report_text)

        f.write(
            "\n\nCONFUSION MATRIX\n"
        )

        f.write(
            "=" * 70 +
            "\n"
        )

        f.write(
            np.array2string(cm)
        )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"JSON: {json_path}"
    )

    print(
        f"TXT:  {txt_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate_resnet()