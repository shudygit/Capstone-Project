"""Evaluation metrics for the global model."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (test_accuracy, average_loss) on the global test set."""
    model.eval()
    model.to(device)
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total
