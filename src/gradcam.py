import os
import sys

import torch
import torch.nn as nn

import numpy as np

from torchvision import transforms

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)


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
        "python src\\gradcam.py "
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

    predicted_class = (
        output.argmax(dim=1).item()
    )


target_layers = [
    model.features[-3]
]


cam = GradCAM(
    model=model,
    target_layers=target_layers
)


targets = [
    ClassifierOutputTarget(
        predicted_class
    )
]


grayscale_cam = cam(
    input_tensor=tensor,
    targets=targets
)[0]


original = np.array(
    image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )
).astype(
    np.float32
) / 255.0


original_rgb = np.stack(
    [
        original,
        original,
        original
    ],
    axis=2
)


visualization = show_cam_on_image(
    original_rgb,
    grayscale_cam,
    use_rgb=True
)


os.makedirs(
    "outputs",
    exist_ok=True
)


output_path = (
    "outputs/gradcam_result.png"
)


Image.fromarray(
    visualization
).save(
    output_path
)


print("=" * 50)
print("GRAD-CAM COMPLETE")
print("=" * 50)

print(
    "Predicted class:",
    classes[predicted_class]
)

print(
    "Saved:",
    output_path
)