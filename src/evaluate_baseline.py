import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


MODEL_PATH = (
    r"models\wafer_cnn_baseline_best.pth"
)

TEST_DIR = (
    r"data\processed\splits\test"
)

DEVICE = torch.device("cpu")

IMAGE_SIZE = 64


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
    batch_size=64,
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
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


true_labels = []
predicted_labels = []


with torch.no_grad():

    for images, labels in loader:

        outputs = model(images)

        predictions = (
            outputs.argmax(dim=1)
        )

        true_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predictions.numpy()
        )


accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


print(
    f"TEST ACCURACY: "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=classes,
        zero_division=0
    )
)