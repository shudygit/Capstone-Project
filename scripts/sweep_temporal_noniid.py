#!/usr/bin/env python
"""Non-IID stress test of the temporal detector.

The temporal detector judges each client by how it leans against the honest
majority over time. Under non-IID data, honest clients also lean persistently in
their own data's direction, so they might be wrongly (and permanently) banned. This
sweep measures that.

For each Dirichlet alpha (large = IID, small = heterogeneous) it runs the temporal
detector in two conditions:
  * no_attack - all honest, so any ban is a FALSE POSITIVE (the number we care about)
  * attack    - the white-box adaptive attacker present, to check detection still holds

It records mean false-positive rate, mean detection rate, final accuracy, and how
many honest clients ended up permanently banned. Writes results/temporal_noniid.csv.

Example:
    python scripts/sweep_temporal_noniid.py --alphas 5.0 1.0 0.3 0.1 0.05 --rounds 12
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.config import load_config  # noqa: E402
from fedblock.server import FederatedServer  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "temporal_hybrid.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-IID stress test of the temporal detector.")
    p.add_argument("--alphas", type=float, nargs="+", default=[5.0, 1.0, 0.3, 0.1, 0.05])
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def run(alpha: float, attack: bool, seed: int, rounds: int, device: str) -> dict:
    cfg = load_config(CONFIG, overrides={
        "experiment": {"seed": seed},
        "data": {"partition": "dirichlet", "dirichlet_alpha": alpha},
        "attack": {"enabled": attack},
        "federated": {"num_rounds": rounds, "device": device},
    })
    server = FederatedServer(cfg)
    hist = pd.DataFrame(server.run(verbose=False))
    honest_banned = sorted(server.temporal_banned - server.malicious)
    return {
        "alpha": alpha,
        "condition": "attack" if attack else "no_attack",
        "seed": seed,
        "final_acc": float(hist["test_acc"].iloc[-1]),
        "mean_detection_rate": float(hist["detection_rate"].mean()),
        "mean_fpr": float(hist["false_positive_rate"].mean()),
        "honest_banned": len(honest_banned),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for seed in args.seeds:
        for alpha in args.alphas:
            for attack in (False, True):
                print(f"seed={seed} alpha={alpha} attack={attack}", flush=True)
                rows.append(run(alpha, attack, seed, args.rounds, args.device))

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "temporal_noniid.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
