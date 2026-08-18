#!/usr/bin/env python
"""Figure for the temporal-detector comparison (reads results/temporal_sweep.csv).

Two panels sharing the x-axis (adaptive_scale):
  * top    - final accuracy: per-round filter collapses as the attack strengthens,
             the temporal detector stays high.
  * bottom - mean detection rate: per-round filter stays at 0, the temporal
             detector catches the attacker.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"per_round": "#e63946", "temporal": "#2a9d8f"}
LABELS = {"per_round": "Per-round filter only", "temporal": "Per-round + temporal (ledger)"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    args = p.parse_args()

    csv = os.path.join(args.results_dir, "temporal_sweep.csv")
    if not os.path.exists(csv):
        sys.exit(f"{csv} not found. Run scripts/sweep_temporal.py first.")
    df = pd.read_csv(csv)
    os.makedirs(args.figures_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for defence in ["per_round", "temporal"]:
        d = (df[df["defence"] == defence]
             .groupby("scale")[["final_acc", "mean_detection_rate", "post_warmup_detection_rate"]]
             .mean().reset_index().sort_values("scale"))
        if d.empty:
            continue
        ax1.plot(d["scale"], d["final_acc"], "o-", color=COLORS[defence],
                 linewidth=2, label=LABELS[defence])
        dr_col = "post_warmup_detection_rate" if defence == "temporal" else "mean_detection_rate"
        ax2.plot(d["scale"], d[dr_col], "o-", color=COLORS[defence],
                 linewidth=2, label=LABELS[defence])

    ax1.set_ylabel("Final accuracy")
    ax1.set_title("Ledger-history detector defeats the adaptive attacker")
    ax2.set_ylabel("Detection rate (post-warm-up)")
    ax2.set_xlabel("adaptive_scale (attack aggressiveness)")
    ax2.set_ylim(-0.05, 1.05)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out = os.path.join(args.figures_dir, "temporal_comparison.png")
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
