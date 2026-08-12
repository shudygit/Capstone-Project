#!/usr/bin/env python
"""Temporal detector vs per-round filter, against the adaptive attacker.

For a range of attack aggressiveness levels, run two defences against the same
white-box adaptive attacker and record final accuracy and mean detection rate:
  * per-round  - the z-score filter only (blind to the adaptive attacker)
  * temporal   - the z-score filter PLUS the ledger-history detector

Writes results/temporal_sweep.csv.

Example:
    python scripts/sweep_temporal.py --scales 0.9 1.5 2.0 3.0 --rounds 12
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
    p = argparse.ArgumentParser(description="Compare per-round vs temporal defence.")
    p.add_argument("--scales", type=float, nargs="+", default=[0.9, 1.5, 2.0, 3.0])
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def run(scale: float, temporal_on: bool, seed: int, rounds: int, device: str) -> dict:
    cfg = load_config(CONFIG, overrides={
        "experiment": {"seed": seed},
        "attack": {"adaptive_scale": scale},
        "defense": {"temporal": temporal_on},     # z-score filter always on
        "federated": {"num_rounds": rounds, "device": device},
    })
    server = FederatedServer(cfg)
    hist = pd.DataFrame(server.run(verbose=False))
    return {
        "defence": "temporal" if temporal_on else "per_round",
        "scale": scale,
        "seed": seed,
        "final_acc": float(hist["test_acc"].iloc[-1]),
        "mean_detection_rate": float(hist["detection_rate"].mean()),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for seed in args.seeds:
        for scale in args.scales:
            for temporal_on in (False, True):
                print(f"seed={seed} scale={scale} temporal={temporal_on}", flush=True)
                rows.append(run(scale, temporal_on, seed, args.rounds, args.device))

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "temporal_sweep.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("Build the figure with: python scripts/make_temporal_figures.py")


if __name__ == "__main__":
    main()
