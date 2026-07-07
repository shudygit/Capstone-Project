"""Federated server: orchestrates FedAvg rounds over simulated clients.

Each round the server:
  1. samples a fraction of clients;
  2. broadcasts the current global model;
  3. collects each client's locally-trained update;
  4. FedAvg-aggregates the updates (weighted by sample count);
  5. evaluates the new global model on the held-out test set.

This is the clean baseline against which the later poisoning-defence modules
are compared.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from .aggregator import fedavg
from .client import Client
from .config import Config
from .data import (load_mnist, make_client_subsets, make_test_loader,
                   partition_data)
from .metrics import evaluate
from .models import build_model
from .utils import clone_state, resolve_device, set_seed


class FederatedServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.experiment.seed)
        self.device = resolve_device(cfg.federated.device)

        train, test = load_mnist(cfg.data.data_root)
        idx = partition_data(train, cfg.data.num_clients, cfg.data.partition,
                             cfg.data.dirichlet_alpha, cfg.experiment.seed)
        shards = make_client_subsets(train, idx)
        self.test_loader = make_test_loader(test, cfg.data.test_batch_size)

        self.clients: List[Client] = [
            Client(cid, shards[cid]) for cid in range(cfg.data.num_clients)
        ]

        self.model = build_model(cfg.model.name).to(self.device)
        self.global_state = clone_state(self.model.state_dict())
        self.history: List[Dict] = []

    def _select_clients(self, round_idx: int) -> List[int]:
        frac = self.cfg.federated.client_fraction
        n = max(1, int(round(frac * self.cfg.data.num_clients)))
        if n >= self.cfg.data.num_clients:
            return list(range(self.cfg.data.num_clients))
        rng = np.random.default_rng(self.cfg.experiment.seed + round_idx)
        return sorted(rng.choice(self.cfg.data.num_clients, size=n, replace=False).tolist())

    def run_round(self, round_idx: int) -> Dict:
        participating = self._select_clients(round_idx)

        t_train = time.perf_counter()
        updates = []
        for cid in participating:
            upd, n = self.clients[cid].local_train(
                self.global_state, self.model, self.cfg.federated,
                self.cfg.data.batch_size, self.device)
            updates.append((upd, n))
        train_time = time.perf_counter() - t_train

        self.global_state = fedavg(updates)
        self.model.load_state_dict(self.global_state)
        acc, loss = evaluate(self.model, self.test_loader, self.device)

        record = {
            "round": round_idx,
            "test_acc": acc,
            "test_loss": loss,
            "num_participating": len(participating),
            "train_time": train_time,
            "round_time": train_time,
        }
        self.history.append(record)
        return record

    def run(self, verbose: bool = True) -> List[Dict]:
        for r in range(self.cfg.federated.num_rounds):
            rec = self.run_round(r)
            if verbose:
                print(f"[baseline] round {r:3d} | acc={rec['test_acc']:.4f} "
                      f"loss={rec['test_loss']:.4f} | t_train={rec['train_time']:.2f}s",
                      flush=True)
        return self.history
