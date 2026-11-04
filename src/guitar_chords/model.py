"""Small CNN baseline; no pretrained weights or downloads required."""

from torch import nn
from . import CHORDS


def build_model():
    layers = []
    channels = 3
    for width in (16, 32, 64):
        layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(.2), nn.Linear(64, len(CHORDS)))
