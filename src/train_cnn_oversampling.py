import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = r"data\processed\splits"
MODEL_DIR = r"models"

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ============================================================
# 3. LOAD DATASETS
# ============================================================

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)


class_names = train_dataset.classes
NUM_CLASSES = len(class_names)


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Classes:", class_names)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

print()


# ============================================================
# 4. CHECK TRAINING CLASS DISTRIBUTION
# ============================================================

train_targets = train_dataset.targets

class_counts = Counter(train_targets)

print("=" * 60)
print("ORIGINAL TRAINING CLASS COUNTS")
print("=" * 60)

for class_index, class_name in enumerate(class_names):

    print(
        f"{class_name:12s}: "
        f"{class_counts[class_index]}"
    )

print()


# ============================================================
# 5. CREATE WEIGHTED RANDOM SAMPLER
# ============================================================

# Calculate inverse-frequency weight for each class.

class_weights = {}

for class_index in range(NUM_CLASSES):

    class_weights[class_index] = (
        1.0 / class_counts[class_index]
    )


# Assign a weight to every training image.

sample_weights = []

for target in train_targets:

    sample_weights.append(
        class_weights[target]
    )


sample_weights = torch.DoubleTensor(sample_weights)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


print("=" * 60)
print("OVERSAMPLING")
print("=" * 60)

print(
    "WeightedRandomSampler created successfully."
)

print(
    "Each training epoch will sample minority classes "
    "more frequently."
)

print()


# ============================================================
# 6. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# 7. CNN MODEL
# ============================================================

class WaferCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = WaferCNN(NUM_CLASSES).to(DEVICE)


print("=" * 60)
print("MODEL")
print("=" * 60)

print(model)

print()

print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)

print()


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

# IMPORTANT:
# We are NOT using class-weighted CrossEntropyLoss here.
#
# Oversampling handles the class imbalance instead.

criterion = nn.CrossEntropyLoss()


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 11. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item() * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        100.0 * correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# 12. VALIDATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item() * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    loss = (
        running_loss / total
    )

    accuracy = (
        100.0 * correct / total
    )


    return loss, accuracy


# ============================================================
# 13. TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0


best_model_path = os.path.join(
    MODEL_DIR,
    "wafer_cnn_oversampling_best.pth"
)


final_model_path = os.path.join(
    MODEL_DIR,
    "wafer_cnn_oversampling_final.pth"
)


print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = evaluate(
        val_loader
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "classes": class_names,

                "best_val_accuracy":
                    best_val_accuracy,

                "image_size":
                    IMAGE_SIZE
            },

            best_model_path
        )


        best_text = " <-- BEST"

    else:

        best_text = ""


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
        f"{best_text}"
    )


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "classes": class_names,

        "final_epoch":
            EPOCHS,

        "image_size":
            IMAGE_SIZE
    },

    final_model_path
)


# ============================================================
# 15. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# 16. TEST BEST MODEL
# ============================================================

test_loss, test_accuracy = evaluate(
    test_loader
)


# ============================================================
# 17. FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Model Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print()

print(
    "Best model saved to:"
)

print(best_model_path)

print()

print(
    "Final model saved to:"
)

print(final_model_path)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)