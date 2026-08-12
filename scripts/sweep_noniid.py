#!/usr/bin/env python
"""Non-IID robustness sweep for the z-score filter.

As clients become more heterogeneous (smaller Dirichlet alpha), honest updates
diverge, so the z-score filter may start flagging honest clients. This sweep runs
the filter across a grid of:

    alpha (data heterogeneity)  x  tau (filter fraction_threshold)

for two conditions:
  * no_attack - all clients honest, so every flag is a FALSE POSITIVE. This
                isolates the false positives caused purely by heterogeneity.
  * attack    - 30% malicious, so we also see how detection holds up.

It writes results/noniid_sweep.csv with one row per (alpha, tau, condition, seed).
Blockchain is left off: it does not affect the filter, and turning it off keeps
the sweep fast.

Example:
    python scripts/sweep_noniid.py --rounds 8 --seeds 0 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.config import load_config  # noqa: E402
from fedblock.server import FederatedServer  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "noniid_hybrid.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep Dirichlet alpha x filter tau.")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[5.0, 1.0, 0.5, 0.1, 0.05])
    p.add_argument("--taus", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def run_one(alpha: float, tau: float, attack: bool, seed: int,
            rounds: int, device: str) -> dict:
    """Run one configuration and summarise the filter's behaviour over the run."""
    cfg = load_config(CONFIG, overrides={
        "experiment": {"seed": seed},
        "data": {"partition": "dirichlet", "dirichlet_alpha": alpha},
        "attack": {"enabled": attack},
        "blockchain": {"enabled": False},
        "defense": {"enabled": True, "fraction_threshold": tau},
        "federated": {"num_rounds": rounds, "device": device},
    })
    server = FederatedServer(cfg)
    hist = pd.DataFrame(server.run(verbose=False))
    return {
        "alpha": alpha,
        "tau": tau,
        "condition": "attack" if attack else "no_attack",
        "seed": seed,
        "final_acc": float(hist["test_acc"].iloc[-1]),
        "mean_detection_rate": float(hist["detection_rate"].mean()),
        "mean_fpr": float(hist["false_positive_rate"].mean()),
        "num_malicious": len(server.malicious),
    }


def main() -> None:
    args = parse_args()
    rows = []
    total = len(args.alphas) * len(args.taus) * 2 * len(args.seeds)
    i = 0
    for seed in args.seeds:
        for alpha in args.alphas:
            for tau in args.taus:
                for attack in (False, True):
                    i += 1
                    print(f"[{i}/{total}] alpha={alpha} tau={tau} "
                          f"attack={attack} seed={seed}", flush=True)
                    rows.append(run_one(alpha, tau, attack, seed,
                                        args.rounds, args.device))

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "noniid_sweep.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("Build the figures with: python scripts/make_noniid_figures.py")


if __name__ == "__main__":
    main()
