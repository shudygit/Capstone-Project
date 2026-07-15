#!/usr/bin/env python
"""Plot accuracy/loss curves from result CSVs.

Overlays every (condition, partition) found in results/, so a baseline run and a
poisoned run of the same partition appear together - making the accuracy
degradation from poisoning directly visible.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "*seed*.csv"))
    if not files:
        sys.exit(f"No result CSVs in {args.results_dir}. Run an experiment first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Backwards-compatibility if an older CSV lacks the 'condition' column.
    if "condition" not in df.columns:
        df["condition"] = "baseline"
    os.makedirs(args.figures_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for (cond, part), sub in df.groupby(["condition", "partition"]):
        label = f"{cond} ({part})"
        acc = sub.groupby("round")["test_acc"].mean()
        ax1.plot(acc.index, acc.values, linewidth=2, label=label)
        loss = sub.groupby("round")["test_loss"].mean()
        ax2.plot(loss.index, loss.values, linewidth=2, label=label)
    ax1.set(xlabel="Communication round", ylabel="Global test accuracy",
            title="FedAvg: baseline vs poisoned accuracy")
    ax2.set(xlabel="Communication round", ylabel="Global test loss",
            title="FedAvg: baseline vs poisoned loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out = os.path.join(args.figures_dir, "accuracy_curves.png")
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
