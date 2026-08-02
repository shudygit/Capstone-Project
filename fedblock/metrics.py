"""Evaluation and detection metrics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Set
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


@dataclass
class DetectionMetrics:
    """How well the Module 4 filter identifies malicious clients in one round.

    Treats "malicious" as the positive class, so the filter is a binary classifier:
      tp = malicious clients correctly flagged   fn = malicious clients missed
      fp = honest clients wrongly flagged         tn = honest clients correctly kept
    """
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def detection_rate(self) -> float:
        """Share of malicious clients caught (recall). tp / (tp + fn)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Share of honest clients wrongly flagged. fp / (fp + tn)."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0


def detection_metrics(flagged: Set[int], malicious: Set[int],
                      participating: Set[int]) -> DetectionMetrics:
    """Confusion counts for one round, restricted to participating clients."""
    flagged = flagged & participating
    mal = malicious & participating
    honest = participating - malicious
    tp = len(flagged & mal)
    fp = len(flagged & honest)
    fn = len(mal - flagged)
    tn = len(honest - flagged)
    return DetectionMetrics(tp=tp, fp=fp, tn=tn, fn=fn)
