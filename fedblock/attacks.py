"""Module 2 - Poisoning attacks: label flipping (data) and gradient-noise (weights).

Two attacks are implemented because they leave *different statistical signatures*,
which matters for the later defence module:

  * label flipping   - the malicious client trains honestly on *mislabelled* data,
                       so its update is a valid-looking but semantically wrong
                       gradient. Hard to spot from weight magnitudes alone.
  * gradient noise   - Gaussian noise (optionally scaled) is added to the final
                       weights, producing a statistically anomalous update.

At this stage there is no defence: these attacks are used to measure how much the
FedAvg baseline degrades under adversarial clients (baseline vulnerability).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch

from torch.utils.data import Dataset

from .config import AttackConfig


class LabelFlipDataset(Dataset):
    """Wraps a dataset and remaps labels for a label-flipping attack.

    Either flips a single source class to a target class, or (if ``flip_all``)
    maps every label ``l -> 9 - l``.
    """

    def __init__(self, base: Dataset, source: int, target: int, flip_all: bool = False):
        self.base = base
        self.source = source
        self.target = target
        self.flip_all = flip_all

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        y = int(y)
        if self.flip_all:
            y = 9 - y
        elif y == self.source:
            y = self.target
        return x, y


def assign_attack_roles(num_clients: int, cfg: AttackConfig, seed: int) -> Dict[int, str]:
    """Decide which clients are malicious and which attack each one runs.

    Returns a dict mapping client_id -> attack_type for malicious clients only
    ("label_flip" or "gradient_noise"); honest clients are absent from the map.
    Malicious clients are split as evenly as possible across the enabled attacks.
    """
    if not cfg.enabled or cfg.malicious_fraction <= 0:
        return {}

    g = torch.Generator().manual_seed(seed)
    num_malicious = int(round(cfg.malicious_fraction * num_clients))
    num_malicious = max(0, min(num_malicious, num_clients))
    perm = torch.randperm(num_clients, generator=g).tolist()
    malicious = perm[:num_malicious]

    types = cfg.types if cfg.types else ["label_flip"]
    roles: Dict[int, str] = {}
    for i, client_id in enumerate(sorted(malicious)):
        roles[client_id] = types[i % len(types)]
    return roles


def apply_gradient_noise(state: Dict[str, torch.Tensor], cfg: AttackConfig,
                         seed: int) -> Dict[str, torch.Tensor]:
    """Return a poisoned copy of ``state`` with Gaussian noise added and optional scaling."""
    g = torch.Generator().manual_seed(seed)
    poisoned = OrderedDict()
    for k, v in state.items():
        v = v.detach().clone().float()
        noise = torch.randn(v.shape, generator=g) * cfg.noise_sigma
        poisoned[k] = (v + noise) * cfg.noise_scale
    return poisoned


def malicious_client_ids(roles: Dict[int, str]) -> List[int]:
    return sorted(roles.keys())
