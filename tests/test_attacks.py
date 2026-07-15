"""Unit tests for module 2 poisoning attacks (no data download needed)."""
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.attacks import (LabelFlipDataset, apply_gradient_noise,
                              assign_attack_roles, malicious_client_ids)
from fedblock.config import AttackConfig


class _ToyDS(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.zeros(1, 28, 28), self.labels[i]


def test_label_flip_single_class():
    ds = LabelFlipDataset(_ToyDS([7, 7, 3, 1]), source=7, target=1)
    labels = [ds[i][1] for i in range(len(ds))]
    assert labels == [1, 1, 3, 1]      # 7 -> 1, others untouched


def test_label_flip_all():
    ds = LabelFlipDataset(_ToyDS([0, 2, 9]), source=0, target=0, flip_all=True)
    labels = [ds[i][1] for i in range(len(ds))]
    assert labels == [9, 7, 0]         # l -> 9 - l


def test_assign_roles_fraction_and_types():
    cfg = AttackConfig(enabled=True, malicious_fraction=0.3,
                       types=["label_flip", "gradient_noise"])
    roles = assign_attack_roles(num_clients=10, cfg=cfg, seed=0)
    assert len(roles) == 3                                   # round(0.3 * 10)
    assert set(roles.values()) <= {"label_flip", "gradient_noise"}
    assert malicious_client_ids(roles) == sorted(roles.keys())


def test_disabled_attack_has_no_malicious():
    cfg = AttackConfig(enabled=False, malicious_fraction=0.5)
    assert assign_attack_roles(10, cfg, seed=0) == {}


def test_gradient_noise_changes_and_is_deterministic():
    state = OrderedDict([("w", torch.ones(100))])
    cfg = AttackConfig(enabled=True, noise_sigma=0.5, noise_scale=1.0)
    a = apply_gradient_noise(state, cfg, seed=42)
    b = apply_gradient_noise(state, cfg, seed=42)
    assert torch.allclose(a["w"], b["w"])                   # reproducible
    assert not torch.allclose(a["w"], state["w"])           # actually perturbed
    assert a["w"].std() > 0.1                                # meaningful noise
