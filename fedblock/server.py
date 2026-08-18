"""Federated server: orchestrates the rounds and the four-scenario logic.

Each round the server:
  1. samples a fraction of clients and broadcasts the global model;
  2. collects each client's locally-trained update (honest or poisoned);
  3. (Module 3, if the blockchain is on) has each client sign its update, verifies
     the signature, records a SHA-256 transaction, and mines a block - timing each
     step so the ledger overhead can be reported;
  4. (Module 4, if the defence is on) runs the z-score filter and drops the
     flagged clients;
  5. FedAvg-averages the surviving updates and evaluates the new global model.

The four scenarios are just combinations of switches in the config:
    baseline           : attack off, blockchain off, defence off
    poisoned_nodefense : attack on,  blockchain off, defence off
    blockchain_only    : attack on,  blockchain on,  defence off
    full_hybrid        : attack on,  blockchain on,  defence on
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from .adaptive_attack import compute_reference, craft_adaptive_update
from .aggregator import fedavg
from .attacks import assign_attack_roles
from .baselines import (fltrust_aggregate, fltrust_scores, iqr_filter,
                        train_root_model)
from .blockchain import (Blockchain, KeyRegistry, LedgerRoundStats, Transaction,
                         sha256_hex, verify_signature)
from .client import Client
from .config import Config
from torch.utils.data import DataLoader, Subset

from .data import (load_mnist, make_client_subsets, make_test_loader,
                   partition_data)
from .defense import zscore_filter
from .metrics import detection_metrics, evaluate
from .models import build_model
from .temporal_detector import deviation_signals, flag_from_ledger
from .utils import clone_state, resolve_device, set_seed, state_to_bytes


def scenario_name(cfg: Config) -> str:
    """Human-readable label for the scenario described by the config switches."""
    if not cfg.attack.enabled:
        return "baseline"
    if not cfg.blockchain.enabled and not cfg.defense.enabled:
        return "poisoned_nodefense"
    if cfg.blockchain.enabled and not cfg.defense.enabled:
        return "blockchain_only"
    # Comparison runs against published defences carry their own label so the
    # results files stay separable from our own hybrid.
    if cfg.defense.method == "iqr":
        return "fedecpa"
    if cfg.defense.method == "fltrust":
        return "fltrust"
    return "full_hybrid"


class FederatedServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scenario = scenario_name(cfg)
        set_seed(cfg.experiment.seed)
        self.device = resolve_device(cfg.federated.device)

        train, test = load_mnist(cfg.data.data_root)
        idx = partition_data(train, cfg.data.num_clients, cfg.data.partition,
                             cfg.data.dirichlet_alpha, cfg.experiment.seed)
        shards = make_client_subsets(train, idx)
        self.test_loader = make_test_loader(test, cfg.data.test_batch_size)

        # Module 2: designate malicious clients and their attack types.
        self.roles = assign_attack_roles(cfg.data.num_clients, cfg.attack,
                                         cfg.experiment.seed)
        self.malicious = set(self.roles.keys())
        self.clients: List[Client] = [
            Client(cid, shards[cid], self.roles.get(cid), cfg.attack)
            for cid in range(cfg.data.num_clients)
        ]

        # Module 3: give every client a signing key and start the ledger.
        self.keys = None
        self.chain = None
        if cfg.blockchain.enabled:
            self.keys = KeyRegistry(cfg.blockchain.rsa_key_bits)
            for cid in range(cfg.data.num_clients):
                self.keys.register(cid)
            self.chain = Blockchain(difficulty=cfg.blockchain.difficulty)

        # FLTrust needs a small clean dataset of its own. We take it from the tail
        # of the training set, deterministically, and never from the test set.
        self.root_client = None
        if cfg.defense.enabled and cfg.defense.method == "fltrust":
            n_root = min(cfg.defense.fltrust_root_size, len(train))
            rng = np.random.default_rng(cfg.experiment.seed + 99991)
            root_idx = rng.choice(len(train), size=n_root, replace=False).tolist()
            self.root_loader = DataLoader(Subset(train, root_idx),
                                          batch_size=cfg.data.batch_size, shuffle=True)
            # Match the server's optimiser steps to an average client's, so the
            # reference update is of comparable magnitude.
            avg_shard = len(train) / cfg.data.num_clients
            self.root_steps = max(1, int(
                cfg.federated.local_epochs * -(-avg_shard // cfg.data.batch_size)))
            self.root_client = True

        self.model = build_model(cfg.model.name).to(self.device)
        self.global_state = clone_state(self.model.state_dict())
        self.history: List[Dict] = []
        # Previous round's honest reference, used by a gray-box adaptive attacker.
        self.prev_ref = None
        # Clients the temporal detector has ever flagged. Ledger evidence only grows,
        # so once a persistent attacker is identified it stays excluded.
        self.temporal_banned: set = set()

    def _select_clients(self, round_idx: int) -> List[int]:
        frac = self.cfg.federated.client_fraction
        n = max(1, int(round(frac * self.cfg.data.num_clients)))
        if n >= self.cfg.data.num_clients:
            return list(range(self.cfg.data.num_clients))
        rng = np.random.default_rng(self.cfg.experiment.seed + round_idx)
        return sorted(rng.choice(self.cfg.data.num_clients, size=n, replace=False).tolist())

    def _record_on_ledger(self, round_idx: int, updates: Dict[int, Dict],
                          sample_counts: Dict[int, int], signals: Dict[int, float]):
        """Sign, verify, hash and mine the round's updates. Returns (stats, accepted).

        'accepted' are the clients whose signature verified; only those go forward.
        Each client's deviation signal is stored on-chain for the temporal detector.
        Timing is split by step so the overhead can be reported.
        """
        stats = LedgerRoundStats()
        txs: List[Transaction] = []
        accepted: List[int] = []

        for cid in sorted(updates):
            # Hash the update.
            t0 = time.perf_counter()
            weight_hash = sha256_hex(state_to_bytes(updates[cid]))
            stats.hash_time += time.perf_counter() - t0

            # Client signs the hash with its private key.
            t0 = time.perf_counter()
            signature = self.keys.register(cid).sign(weight_hash.encode())
            stats.sign_time += time.perf_counter() - t0

            # Server verifies the signature with the client's public key.
            t0 = time.perf_counter()
            ok = verify_signature(self.keys.public_key(cid), weight_hash.encode(), signature)
            stats.verify_time += time.perf_counter() - t0

            txs.append(Transaction(cid, round_idx, weight_hash, sample_counts[cid],
                                   signature.hex(), accepted=ok,
                                   deviation_signal=signals.get(cid, 0.0)))
            if ok:
                accepted.append(cid)
            else:
                stats.num_rejected += 1

        # Mine the block that seals this round's transactions.
        t0 = time.perf_counter()
        self.chain.add_block(txs)
        stats.mine_time += time.perf_counter() - t0
        stats.num_transactions = len(txs)
        return stats, accepted

    def _collect_updates(self, participating: List[int], round_idx: int):
        """Gather each client's update. Adaptive attackers are handled in a second
        phase because they need the honest clients' statistics to craft with."""
        cfg = self.cfg
        adaptive_ids = [c for c in participating if self.roles.get(c) == "adaptive"]

        updates: Dict[int, Dict] = {}
        sample_counts: Dict[int, int] = {}

        # Phase 1: everyone except adaptive attackers trains normally.
        for cid in participating:
            if cid in adaptive_ids:
                continue
            upd, n = self.clients[cid].local_train(
                self.global_state, self.model, cfg.federated,
                cfg.data.batch_size, self.device, round_idx)
            updates[cid] = upd
            sample_counts[cid] = n

        # Phase 2: adaptive attackers craft filter-evading updates.
        if adaptive_ids:
            honest_ids = [c for c in participating if c not in self.malicious]
            current_ref = (compute_reference(updates, honest_ids)
                           if len(honest_ids) >= 2 else None)
            # White-box uses the current round's stats; gray-box uses last round's.
            if cfg.attack.adaptive_mode == "graybox":
                ref = self.prev_ref if self.prev_ref is not None else current_ref
            else:
                ref = current_ref
            for cid in adaptive_ids:
                if ref is None:                     # not enough info yet: act honestly
                    upd, n = self.clients[cid].local_train(
                        self.global_state, self.model, cfg.federated,
                        cfg.data.batch_size, self.device, round_idx)
                else:
                    upd = craft_adaptive_update(ref, self.global_state,
                                                cfg.defense.z_threshold,
                                                cfg.attack.adaptive_scale)
                    n = self.clients[cid].num_samples
                updates[cid] = upd
                sample_counts[cid] = n
            if current_ref is not None:
                self.prev_ref = current_ref         # remember for a gray-box attacker
        return updates, sample_counts

    def run_round(self, round_idx: int) -> Dict:
        cfg = self.cfg
        participating = self._select_clients(round_idx)

        # 1) Local training -------------------------------------------------
        t_train = time.perf_counter()
        updates, sample_counts = self._collect_updates(participating, round_idx)
        train_time = time.perf_counter() - t_train

        # Per-client deviation signals (used by the ledger + temporal detector).
        signals = deviation_signals(updates, self.global_state)

        # 2) Blockchain logging (Module 3) ----------------------------------
        blockchain_time = 0.0
        candidates = list(updates.keys())
        if cfg.blockchain.enabled:
            stats, accepted = self._record_on_ledger(round_idx, updates,
                                                     sample_counts, signals)
            blockchain_time = stats.total_time
            candidates = accepted            # only signature-valid updates continue

        # 3) Defence (Module 4) 

        flagged: List[int] = []
        defense_time = 0.0
        trust_scores = None
        server_state = None
        if cfg.defense.enabled:
            t0 = time.perf_counter()
            method = cfg.defense.method
            if method == "zscore":
                flagged = zscore_filter(
                    {c: updates[c] for c in candidates},
                    z_threshold=cfg.defense.z_threshold,
                    fraction_threshold=cfg.defense.fraction_threshold).flagged
            elif method == "iqr":
                flagged = iqr_filter(
                    {c: updates[c] for c in candidates},
                    k=cfg.defense.iqr_k,
                    fraction_threshold=cfg.defense.fraction_threshold).flagged
            elif method == "fltrust":
                # The server trains on its own clean root data to get a reference
                # direction, then scores every client against it.
                server_state = train_root_model(
                    self.global_state, self.model, cfg.federated,
                    self.root_loader, self.device, self.root_steps)
                trust_scores = fltrust_scores(
                    {c: updates[c] for c in candidates},
                    self.global_state, server_state)
                # A zero trust score means the client is excluded outright, which is
                # the same decision the filters make, so it counts as a flag.
                flagged = [c for c in candidates if trust_scores.get(c, 0.0) <= 0.0]
            else:
                raise ValueError(f"Unknown defense method '{method}'")
            defense_time = time.perf_counter() - t0

        # 3b) Temporal detector (novelty) - reads the ledger history (Module 5)
        if cfg.defense.temporal and self.chain is not None:
            t0 = time.perf_counter()
            temporal_flagged = flag_from_ledger(
                self.chain, warmup=cfg.defense.temporal_warmup,
                threshold=cfg.defense.temporal_threshold)
            defense_time += time.perf_counter() - t0
            # Bans are sticky: ledger evidence never disappears, so a client caught
            # once stays caught. A client flagged by either detector is dropped.
            self.temporal_banned |= set(temporal_flagged)
            flagged = sorted(set(flagged) | (self.temporal_banned & set(candidates)))

        keep = [c for c in candidates if c not in set(flagged)]
        if not keep:                          # safety net: never drop everyone
            keep = candidates

        # 4) Aggregate + evaluate
        if trust_scores is not None:
            # FLTrust does not average equally: it weights by trust and rescales
            # every update to the server's own magnitude.
            self.global_state = fltrust_aggregate(
                {c: updates[c] for c in keep}, self.global_state, server_state,
                {c: trust_scores.get(c, 0.0) for c in keep})
        else:
            self.global_state = fedavg([(updates[c], sample_counts[c]) for c in keep])
        self.model.load_state_dict(self.global_state)
        acc, loss = evaluate(self.model, self.test_loader, self.device)

        # 5) Detection metrics 
        det = detection_metrics(set(flagged), self.malicious, set(participating))

        record = {
            "round": round_idx,
            "test_acc": acc,
            "test_loss": loss,
            "num_participating": len(participating),
            "num_malicious_participating": len(set(participating) & self.malicious),
            "num_flagged": len(flagged),
            "num_aggregated": len(keep),
            "detection_rate": det.detection_rate,
            "false_positive_rate": det.false_positive_rate,
            "tp": det.tp, "fp": det.fp, "tn": det.tn, "fn": det.fn,
            "train_time": train_time,
            "blockchain_time": blockchain_time,
            "defense_time": defense_time,
            "round_time": train_time + blockchain_time + defense_time,
        }
        self.history.append(record)
        return record

    def run(self, verbose: bool = True) -> List[Dict]:
        if verbose and self.malicious:
            print(f"Malicious clients: {sorted(self.malicious)} "
                  f"({ {c: self.roles[c] for c in sorted(self.malicious)} })", flush=True)
        for r in range(self.cfg.federated.num_rounds):
            rec = self.run_round(r)
            if verbose:
                print(f"[{self.scenario}] round {r:3d} | acc={rec['test_acc']:.4f} "
                      f"loss={rec['test_loss']:.4f} | flagged={rec['num_flagged']} "
                      f"(DR={rec['detection_rate']:.2f} FPR={rec['false_positive_rate']:.2f}) "
                      f"| t_train={rec['train_time']:.2f}s t_chain={rec['blockchain_time']:.3f}s",
                      flush=True)
        # A tampered or broken chain must never pass silently.
        if self.chain is not None and not self.chain.is_valid():
            raise RuntimeError("Blockchain integrity check failed after the run!")
        return self.history
