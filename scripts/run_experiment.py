#!/usr/bin/env python
"""Run the FedAvg baseline and save per-round results.

Examples:
    python scripts/run_experiment.py --config configs/baseline.yaml --seed 0
    python scripts/run_experiment.py --config configs/baseline.yaml \
        --set data.partition=dirichlet --set data.dirichlet_alpha=0.1
    python scripts/run_experiment.py --quick        # fast smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.config import load_config  # noqa: E402
from fedblock.server import FederatedServer  # noqa: E402


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the FedAvg baseline.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: 3 rounds, 6 clients.")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="section.key=value",
                   help="Generic override, e.g. --set data.partition=dirichlet (repeatable).")
    return p.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    o: dict = {}
    if args.seed is not None:
        o.setdefault("experiment", {})["seed"] = args.seed
    if args.results_dir is not None:
        o.setdefault("experiment", {})["results_dir"] = args.results_dir
    if args.rounds is not None:
        o.setdefault("federated", {})["num_rounds"] = args.rounds
    if args.device is not None:
        o.setdefault("federated", {})["device"] = args.device
    if args.quick:
        o.setdefault("federated", {})["num_rounds"] = args.rounds or 3
        o.setdefault("data", {})["num_clients"] = 6
    for item in args.overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(f"--set expects 'section.key=value', got '{item}'")
        path, raw = item.split("=", 1)
        section, key = path.split(".", 1)
        o.setdefault(section, {})[key] = _coerce(raw)
    return o


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=build_overrides(args))

    kind = "poisoned FedAvg" if cfg.attack.enabled else "FedAvg baseline"
    print(f"=== {kind} (seed={cfg.experiment.seed}) ===")
    print(f"    clients={cfg.data.num_clients} partition={cfg.data.partition} "
          f"rounds={cfg.federated.num_rounds} model={cfg.model.name}")
    if cfg.attack.enabled:
        print(f"    attack: types={cfg.attack.types} "
              f"malicious_fraction={cfg.attack.malicious_fraction} "
              f"noise_sigma={cfg.attack.noise_sigma}")

    t0 = time.time()
    server = FederatedServer(cfg)
    history = server.run(verbose=not args.quiet)
    wall = time.time() - t0

    results_dir = cfg.experiment.results_dir
    os.makedirs(results_dir, exist_ok=True)
    kind_tag = "poisoned" if cfg.attack.enabled else "baseline"
    tag = f"{kind_tag}_{cfg.data.partition}_seed{cfg.experiment.seed}"

    df = pd.DataFrame(history)
    df.insert(0, "condition", kind_tag)
    df.insert(1, "partition", cfg.data.partition)
    df.insert(2, "seed", cfg.experiment.seed)
    csv_path = os.path.join(results_dir, f"{tag}.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "config": cfg.to_dict(),
        "condition": kind_tag,
        "malicious_clients": sorted(server.malicious),
        "final_acc": float(df["test_acc"].iloc[-1]),
        "best_acc": float(df["test_acc"].max()),
        "mean_train_time": float(df["train_time"].mean()),
        "wall_seconds": wall,
    }
    with open(os.path.join(results_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinished in {wall:.1f}s | final acc={summary['final_acc']:.4f} "
          f"| best acc={summary['best_acc']:.4f}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
