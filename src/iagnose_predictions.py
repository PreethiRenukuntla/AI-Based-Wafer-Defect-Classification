import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader

from src.config import BEST_MODEL_PATH, DEFECT_CLASSES, DEVICE
from src.data.dataset import get_data_splits
from src.models.architecture import WaferDefectResNet


print("=" * 70)
print("RESNET OFFICIAL TEST PREDICTION DISTRIBUTION")
print("=" * 70)

device = torch.device(DEVICE)

print("\nLoading official test dataset...")

_, _, _, _, test_dataset = get_data_splits(
    val_ratio=0.15,
    seed=42
)

print(f"Official test samples: {len(test_dataset):,}")

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

print("\nLoading ResNet model...")

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device
)

model = WaferDefectResNet(
    num_classes=9,
    in_channels=1
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")
print("\nRunning predictions...")
print("Please wait. Do NOT start training.")


true_labels = []
pred_labels = []

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        ).cpu().numpy()

        pred_labels.extend(predictions)
        true_labels.extend(labels.numpy())

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Processed batches: "
                f"{batch_idx + 1}/{len(test_loader)}"
            )


true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)


print()
print("=" * 70)
print("TRUE TEST DISTRIBUTION")
print("=" * 70)

true_counts = Counter(true_labels)

for idx, name in enumerate(DEFECT_CLASSES):

    count = true_counts[idx]

    percentage = (
        count / len(true_labels) * 100
    )

    print(
        f"{name:12s}: "
        f"{count:7,d} "
        f"({percentage:6.2f}%)"
    )


print()
print("=" * 70)
print("PREDICTED TEST DISTRIBUTION")
print("=" * 70)

pred_counts = Counter(pred_labels)

for idx, name in enumerate(DEFECT_CLASSES):

    count = pred_counts[idx]

    percentage = (
        count / len(pred_labels) * 100
    )

    print(
        f"{name:12s}: "
        f"{count:7,d} "
        f"({percentage:6.2f}%)"
    )


print()
print("=" * 70)
print("PREDICTION / TRUE RATIO")
print("=" * 70)

for idx, name in enumerate(DEFECT_CLASSES):

    true_count = true_counts[idx]
    pred_count = pred_counts[idx]

    ratio = (
        pred_count / true_count
        if true_count > 0
        else 0
    )

    print(
        f"{name:12s}: "
        f"true={true_count:7,d} "
        f"pred={pred_count:7,d} "
        f"ratio={ratio:6.2f}"
    )


print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)