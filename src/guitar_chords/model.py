"""Models for chord classification."""

from torch import nn
from . import CHORDS

RESNET18 = "resnet18"
RESNET18_PREPROCESSING = "ResNet18_Weights.IMAGENET1K_V1"


def build_resnet18(*, pretrained=False):
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if pretrained:
        model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(CHORDS))
    return model


def build_model():
    layers = []
    channels = 3
    for width in (16, 32, 64):
        layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(.2), nn.Linear(64, len(CHORDS)))
