"""Federated client: local SGD training, with optional poisoning behaviour."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .attacks import LabelFlipDataset, apply_gradient_noise
from .config import AttackConfig, FederatedConfig
from .utils import clone_state


class Client:
    """One simulated federated client owning a private data shard.

    An honest client has ``attack_role=None``. A malicious client runs either
    "label_flip" (trains on mislabelled data) or "gradient_noise" (perturbs its
    final weights).
    """

    def __init__(self, client_id: int, dataset: Subset,
                 attack_role: Optional[str] = None,
                 attack_cfg: Optional[AttackConfig] = None):
        self.client_id = client_id
        self.dataset = dataset
        self.attack_role = attack_role
        self.attack_cfg = attack_cfg

    @property
    def is_malicious(self) -> bool:
        return self.attack_role is not None

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def _train_dataset(self):
        """Return the (possibly poisoned) dataset used for this client's training."""
        if self.attack_role == "label_flip" and self.attack_cfg is not None:
            return LabelFlipDataset(
                self.dataset,
                source=self.attack_cfg.label_flip_source,
                target=self.attack_cfg.label_flip_target,
                flip_all=self.attack_cfg.label_flip_all,
            )
        return self.dataset

    def local_train(self, global_state: Dict[str, torch.Tensor], model: nn.Module,
                    fed_cfg: FederatedConfig, batch_size: int, device: torch.device,
                    round_idx: int = 0) -> Tuple[Dict[str, torch.Tensor], int]:
        """Run local SGD from the broadcast global weights and return the update.

        Malicious "label_flip" clients train on mislabelled data; "gradient_noise"
        clients train honestly then perturb the final weights.
        """
        model.load_state_dict(global_state)
        model.to(device)
        model.train()

        loader = DataLoader(self._train_dataset(), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=fed_cfg.lr,
                                    momentum=fed_cfg.momentum)

        for _ in range(fed_cfg.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

        update = clone_state(model.state_dict())

        if self.attack_role == "gradient_noise" and self.attack_cfg is not None:
            # Unique-but-deterministic seed per client/round for reproducibility.
            seed = 1_000_003 * (round_idx + 1) + self.client_id
            update = apply_gradient_noise(update, self.attack_cfg, seed)

        return update, self.num_samples
