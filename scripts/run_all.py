#!/usr/bin/env python
"""Run all four scenarios (optionally across several seeds), then remind how to
build the figures and the summary table.

Examples:
    python scripts/run_all.py                 # all 4 scenarios, seed 0
    python scripts/run_all.py --seeds 0 1 2   # 3 seeds each
    python scripts/run_all.py --quick         # fast smoke of all 4 scenarios
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCENARIOS = [
    "configs/baseline.yaml",
    "configs/poisoned.yaml",
    "configs/blockchain_only.yaml",
    "configs/full_hybrid.yaml",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all four fedblock scenarios.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for scenario in SCENARIOS:
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_experiment.py"),
                   "--config", scenario, "--seed", str(seed),
                   "--results-dir", args.results_dir]
            if args.rounds is not None:
                cmd += ["--rounds", str(args.rounds)]
            if args.device is not None:
                cmd += ["--device", args.device]
            if args.quick:
                # Fast smoke: few rounds and low Proof-of-Work difficulty.
                cmd += ["--quick", "--set", "blockchain.difficulty=2"]
            print("\n>>> " + " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=ROOT)

    print("\nAll scenarios complete. Now build the outputs:")
    print("    python scripts/make_figures.py")
    print("    python scripts/make_tables.py")


if __name__ == "__main__":
    main()
