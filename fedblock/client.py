"""Federated client: local SGD training on a private data shard."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import FederatedConfig
from .utils import clone_state


class Client:
    """One simulated federated client owning a private data shard."""

    def __init__(self, client_id: int, dataset: Subset):
        self.client_id = client_id
        self.dataset = dataset

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def local_train(self, global_state: Dict[str, torch.Tensor], model: nn.Module,
                    fed_cfg: FederatedConfig, batch_size: int, device: torch.device
                    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Run local SGD from the broadcast global weights and return the update."""
        model.load_state_dict(global_state)
        model.to(device)
        model.train()

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=fed_cfg.lr,
                                    momentum=fed_cfg.momentum)

        for _ in range(fed_cfg.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

        return clone_state(model.state_dict()), self.num_samples
