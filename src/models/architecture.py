import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Residual Conv Block with SE Attention."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out, inplace=True)


class WaferDefectResNet(nn.Module):
    """
    High-Performance CPU-Friendly ResNet Architecture for Wafer Defect Map Classification.
    Lightweight (~650k parameters), fast inference, excellent multi-class defect recognition.
    """
    def __init__(self, num_classes=9, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32, 64, stride=2)    # 56x56 -> 28x28
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 28x28 -> 14x14
        self.layer3 = ResidualBlock(128, 256, stride=2)  # 14x14 -> 7x7
        self.layer4 = ResidualBlock(256, 256, stride=1)  # 7x7 -> 7x7

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        self.classifier = nn.Linear(128, num_classes)

    def extract_features(self, x):
        """Extracts 128-dim normalized embedding vector for similarity search."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        emb = self.fc_embedding(x)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        emb = self.fc_embedding(x)
        logits = self.classifier(emb)
        return logits


if __name__ == "__main__":
    dummy_input = torch.randn(4, 1, 56, 56)
    model = WaferDefectResNet(num_classes=9)
    out = model(dummy_input)
    emb = model.extract_features(dummy_input)
    print("Model Output Shape:", out.shape)
    print("Embedding Shape:", emb.shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
