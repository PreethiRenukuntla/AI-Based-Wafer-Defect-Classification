import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATA_DIR = r"data\processed\splits"
MODEL_DIR = r"models"

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


test_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


class WaferCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
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

        return self.classifier(
            self.features(x)
        )


model = WaferCNN(
    len(train_dataset.classes)
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_accuracy = 0


best_path = (
    r"models\wafer_cnn_baseline_best.pth"
)

final_path = (
    r"models\wafer_cnn_baseline_final.pth"
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
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

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = total_loss / total

    train_accuracy = (
        correct / total * 100
    )


    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_accuracy = (
        val_correct / val_total * 100
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_accuracy:.2f}% | "
        f"Val Acc {val_accuracy:.2f}%"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "best_val_accuracy":
                    best_val_accuracy,

                "image_size":
                    IMAGE_SIZE
            },
            best_path
        )


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "classes":
            train_dataset.classes,

        "image_size":
            IMAGE_SIZE
    },
    final_path
)


print("\nTraining complete.")
print(
    "Best validation accuracy:",
    best_val_accuracy
)