"""Shared utilities: seeding and device selection."""
from __future__ import annotations

import random
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    "auto" prefers CUDA, then Apple MPS, then CPU.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep-copy a state_dict onto CPU (detached)."""
    return OrderedDict((k, v.detach().clone().cpu()) for k, v in state.items())
