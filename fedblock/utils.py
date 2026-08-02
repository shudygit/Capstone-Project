"""Shared utilities: seeding, device selection, and weight (de)serialisation."""
from __future__ import annotations

import io
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


def state_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a model's weights into one long 1-D vector (keys sorted).

    Sorting the keys guarantees the same layout every time, so the z-score filter
    (Module 4) always compares clients coordinate-for-coordinate.
    """
    parts = [state[k].detach().reshape(-1).float().cpu() for k in sorted(state.keys())]
    return torch.cat(parts)


def state_to_bytes(state: Dict[str, torch.Tensor]) -> bytes:
    """Serialise a model's weights to deterministic bytes for SHA-256 hashing.

    Weights are moved to CPU and saved with a fixed key order, so the same model
    always produces the same byte stream (and therefore the same hash) - which is
    what the blockchain ledger (Module 3) relies on.
    """
    ordered = OrderedDict((k, state[k].detach().cpu()) for k in sorted(state.keys()))
    buf = io.BytesIO()
    torch.save(ordered, buf)
    return buf.getvalue()
