#!/usr/bin/env python
"""Figure for the non-IID stress test of the temporal detector.

Reads results/temporal_noniid.csv and draws two panels sharing the x-axis (alpha):
  * top    - honest-only false-positive rate vs heterogeneity (the detector wrongly
             bans honest clients as data gets more non-IID).
  * bottom - detection rate and final accuracy under attack vs heterogeneity (the
             detector stops working under strong non-IID).
"""
from __future__ import annotations

import argparse
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

    csv = os.path.join(args.results_dir, "temporal_noniid.csv")
    if not os.path.exists(csv):
        sys.exit(f"{csv} not found. Run scripts/sweep_temporal_noniid.py first.")
    df = pd.read_csv(csv)
    os.makedirs(args.figures_dir, exist_ok=True)

    honest = (df[df["condition"] == "no_attack"]
              .groupby("alpha")["mean_fpr"].mean().reset_index().sort_values("alpha"))
    attack = (df[df["condition"] == "attack"]
              .groupby("alpha")[["mean_detection_rate", "final_acc"]]
              .mean().reset_index().sort_values("alpha"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(honest["alpha"], honest["mean_fpr"], "o-", color="#e63946", linewidth=2)
    ax1.set_ylabel("Honest-only false-positive rate")
    ax1.set_title("The temporal detector inherits the non-IID limitation")

    ax2.plot(attack["alpha"], attack["mean_detection_rate"], "o-",
             color="#264653", linewidth=2, label="detection rate")
    ax2.plot(attack["alpha"], attack["final_acc"], "s--",
             color="#2a9d8f", linewidth=2, label="final accuracy (under attack)")
    ax2.set_ylabel("Rate / accuracy")
    ax2.set_xlabel("Dirichlet alpha (log scale; right = more non-IID)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(args.figures_dir, "temporal_noniid.png")
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
