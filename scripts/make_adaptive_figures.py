#!/usr/bin/env python
"""Figure for the adaptive-attacker sweep (reads results/adaptive_sweep.csv).

Produces adaptive_frontier.png: two panels sharing the x-axis (adaptive_scale).
  * top    - detection rate vs scale, white-box and gray-box. Shows the filter
             catching nothing below scale 1 and catching the attacker above it.
  * bottom - final accuracy vs scale, with the clean (no-attack) reference line.
             Shows the damage the attacker achieves while it remains undetected.
A vertical line at scale = 1 marks the filter's detection boundary.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"whitebox": "#264653", "graybox": "#e76f51"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    args = p.parse_args()

    csv = os.path.join(args.results_dir, "adaptive_sweep.csv")
    if not os.path.exists(csv):
        sys.exit(f"{csv} not found. Run scripts/sweep_adaptive.py first.")
    df = pd.read_csv(csv)
    os.makedirs(args.figures_dir, exist_ok=True)

    clean_acc = df[df["mode"] == "clean"]["final_acc"].mean()
    attack = df[df["mode"] != "clean"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for mode in ["whitebox", "graybox"]:
        d = (attack[attack["mode"] == mode]
             .groupby("scale")[["mean_detection_rate", "final_acc"]]
             .mean().reset_index().sort_values("scale"))
        if d.empty:
            continue
        ax1.plot(d["scale"], d["mean_detection_rate"], "o-",
                 color=COLORS[mode], linewidth=2, label=mode)
        ax2.plot(d["scale"], d["final_acc"], "o-",
                 color=COLORS[mode], linewidth=2, label=mode)

    for ax in (ax1, ax2):
        # scale = 1 is where the crafted |z| equals the filter threshold *relative to
        # the honest cohort*; note the filter still never flags, because the attackers
        # inflate the shared std it uses to judge them.
        ax.axvline(1.0, color="grey", linestyle="--", alpha=0.7)
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.text(1.03, 0.5, "crafted |z| = threshold\n(vs honest cohort)",
             color="grey", fontsize=8, va="center")
    ax1.set_ylabel("Mean detection rate")
    ax1.set_ylim(-0.05, 1.02)
    ax1.set_title("Adaptive attacker: the filter never detects it (top);\n"
                  "damage depends on the attacker's knowledge (bottom)")

    ax2.axhline(clean_acc, color="#2a9d8f", linestyle=":", linewidth=2,
                label=f"clean (no attack) = {clean_acc:.3f}")
    ax2.set_ylabel("Final accuracy")
    ax2.set_xlabel("adaptive_scale  (fraction of the filter threshold)")
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(args.figures_dir, "adaptive_frontier.png")
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
