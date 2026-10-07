import sys

import torch
import torch.nn as nn

from torchvision import transforms

from PIL import Image


MODEL_PATH = (
    r"models\wafer_cnn_baseline_best.pth"
)

IMAGE_SIZE = 64

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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


if len(sys.argv) < 2:

    print(
        "Usage:"
    )

    print(
        "python src\\predict.py "
        "path_to_image.png"
    )

    sys.exit()


image_path = sys.argv[1]


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


image = Image.open(
    image_path
).convert("L")


tensor = transform(
    image
).unsqueeze(0).to(DEVICE)


with torch.no_grad():

    output = model(tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    predicted_index = (
        probabilities.argmax().item()
    )


print("=" * 50)
print("WAFER DEFECT PREDICTION")
print("=" * 50)

print(
    "Predicted Class:",
    classes[predicted_index]
)

print(
    "Confidence:",
    f"{probabilities[predicted_index].item() * 100:.2f}%"
)

print()

print("Class Probabilities")

for i, class_name in enumerate(classes):

    print(
        f"{class_name:12s}: "
        f"{probabilities[i].item() * 100:.2f}%"
    )