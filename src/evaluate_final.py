import os

import torch
import torch.nn as nn

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


MODEL_PATH = (
    r"models\wafer_cnn_baseline_best.pth"
)

TEST_DIR = (
    r"data\processed\splits\test"
)

IMAGE_SIZE = 64
BATCH_SIZE = 64


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


transform = transforms.Compose([

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


dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)


loader = DataLoader(
    dataset,
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


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


classes = checkpoint["classes"]


model = WaferCNN(
    len(classes)
).to(DEVICE)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


true_labels = []
predicted_labels = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = (
            outputs.argmax(dim=1)
            .cpu()
            .numpy()
        )

        predicted_labels.extend(
            predictions
        )

        true_labels.extend(
            labels.numpy()
        )


accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


print("=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()

print("Classification Report")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=classes,
        zero_division=0
    )
)


cm = confusion_matrix(
    true_labels,
    predicted_labels
)


print("Confusion Matrix")
print("=" * 60)

print(cm)


os.makedirs(
    "outputs",
    exist_ok=True
)


with open(
    "outputs/final_results.txt",
    "w"
) as file:

    file.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=classes,
            zero_division=0
        )
    )


print()
print(
    "Saved: outputs/final_results.txt"
)