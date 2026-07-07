"""Model definitions for MNIST classification."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """A compact 2-conv + 2-FC CNN that reaches ~99% clean test accuracy on MNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class MLP(nn.Module):
    """A lightweight 2-layer fully-connected baseline (~97% clean accuracy)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name == "small_cnn":
        return SmallCNN(num_classes)
    if name == "mlp":
        return MLP(num_classes)
    raise ValueError(f"Unknown model '{name}' (expected 'small_cnn' or 'mlp')")
