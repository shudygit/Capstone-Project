#!/usr/bin/env python
"""Adaptive-attacker sweep: the evasion vs damage frontier.

An adaptive attacker crafts updates bounded to |z| = adaptive_scale * z_threshold.
With scale < 1 it stays under the filter and is never flagged; as scale crosses 1
its weights become outliers and the filter starts catching (and removing) it.

This sweep varies adaptive_scale for both threat models (white-box and gray-box)
and records, per configuration:
  * detection_rate - how often the filter catches the attacker (should be ~0 below 1)
  * final accuracy  - the model's accuracy under the attack
so we can plot where the attacker evades and how much damage it achieves.

A clean-baseline reference (no attack) is included so 'damage' can be read off.

Example:
    python scripts/sweep_adaptive.py --scales 0.5 0.8 0.9 1.0 1.2 --rounds 10 --seeds 0 1
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.config import load_config  # noqa: E402
from fedblock.server import FederatedServer  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "adaptive_hybrid.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep the adaptive attacker's aggressiveness.")
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.5, 0.7, 0.9, 1.0, 1.2, 1.5])
    p.add_argument("--modes", nargs="+", default=["whitebox", "graybox"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def run(overrides: dict) -> pd.DataFrame:
    cfg = load_config(CONFIG, overrides=overrides)
    server = FederatedServer(cfg)
    return pd.DataFrame(server.run(verbose=False))


def main() -> None:
    args = parse_args()
    rows = []

    for seed in args.seeds:
        # Clean reference: defence on, no attack, so we know the undamaged accuracy.
        base = run({"experiment": {"seed": seed}, "attack": {"enabled": False},
                    "federated": {"num_rounds": args.rounds, "device": args.device}})
        rows.append({"mode": "clean", "scale": None, "seed": seed,
                     "final_acc": float(base["test_acc"].iloc[-1]),
                     "mean_detection_rate": 0.0})

        for mode in args.modes:
            for scale in args.scales:
                print(f"seed={seed} mode={mode} scale={scale}", flush=True)
                hist = run({
                    "experiment": {"seed": seed},
                    "attack": {"enabled": True, "adaptive_scale": scale,
                               "adaptive_mode": mode},
                    "federated": {"num_rounds": args.rounds, "device": args.device},
                })
                rows.append({"mode": mode, "scale": scale, "seed": seed,
                             "final_acc": float(hist["test_acc"].iloc[-1]),
                             "mean_detection_rate": float(hist["detection_rate"].mean())})

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "adaptive_sweep.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("Build the figure with: python scripts/make_adaptive_figures.py")


if __name__ == "__main__":
    main()
