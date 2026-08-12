#!/usr/bin/env python
"""Figures for the non-IID robustness sweep (reads results/noniid_sweep.csv).

Produces:
  * noniid_fpr_vs_alpha.png  - false-positive rate vs heterogeneity, at the default
    tau, for the honest-only (no_attack) and attacked conditions. Shows the filter
    wrongly flagging honest clients as data gets more non-IID.
  * noniid_detection_vs_alpha.png - detection rate vs heterogeneity under attack.
  * noniid_fpr_heatmap.png   - false-positive rate over (alpha, tau) for the
    honest-only condition, marking where the filter stays safe.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TAU = 0.05


def _save(fig_path: str) -> None:
    plt.savefig(fig_path, dpi=200)
    plt.savefig(fig_path.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved {fig_path}")


def _mean_over_seeds(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (df.groupby(["alpha", "tau", "condition"])[col]
              .mean().reset_index())


def plot_fpr_vs_alpha(df: pd.DataFrame, out: str, tau: float) -> None:
    sub = _mean_over_seeds(df[df["tau"] == tau], "mean_fpr")
    plt.figure(figsize=(8, 5))
    for cond, color in [("no_attack", "#2a9d8f"), ("attack", "#e63946")]:
        d = sub[sub["condition"] == cond].sort_values("alpha")
        if not d.empty:
            label = "Honest only (pure false positives)" if cond == "no_attack" \
                else "Under attack"
            plt.plot(d["alpha"], d["mean_fpr"], "o-", color=color, linewidth=2, label=label)
    plt.xscale("log")
    plt.gca().invert_xaxis()  # left = more IID, right = more heterogeneous
    plt.xlabel("Dirichlet alpha (log scale; right = more non-IID)")
    plt.ylabel("Mean false-positive rate")
    plt.title(f"Filter false positives rise with heterogeneity (tau={tau})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save(out)


def plot_detection_vs_alpha(df: pd.DataFrame, out: str, tau: float) -> None:
    sub = _mean_over_seeds(df[(df["tau"] == tau) & (df["condition"] == "attack")],
                           "mean_detection_rate").sort_values("alpha")
    if sub.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(sub["alpha"], sub["mean_detection_rate"], "o-",
             color="#264653", linewidth=2)
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Dirichlet alpha (log scale; right = more non-IID)")
    plt.ylabel("Mean detection rate")
    plt.title(f"Detection rate vs heterogeneity under attack (tau={tau})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(out)


def plot_fpr_heatmap(df: pd.DataFrame, out: str) -> None:
    sub = _mean_over_seeds(df[df["condition"] == "no_attack"], "mean_fpr")
    if sub.empty:
        return
    alphas = sorted(sub["alpha"].unique(), reverse=True)   # rows: more non-IID at bottom
    taus = sorted(sub["tau"].unique())
    grid = np.full((len(alphas), len(taus)), np.nan)
    for r, a in enumerate(alphas):
        for c, t in enumerate(taus):
            cell = sub[(sub["alpha"] == a) & (sub["tau"] == t)]
            if not cell.empty:
                grid[r, c] = cell["mean_fpr"].iloc[0]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(grid, aspect="auto", cmap="Reds", vmin=0.0)
    plt.colorbar(im, label="Mean false-positive rate (honest only)")
    plt.xticks(range(len(taus)), [str(t) for t in taus])
    plt.yticks(range(len(alphas)), [str(a) for a in alphas])
    plt.xlabel("Filter tau (fraction_threshold)")
    plt.ylabel("Dirichlet alpha (top = IID, bottom = non-IID)")
    plt.title("False-positive rate over (alpha, tau)")
    for r in range(len(alphas)):
        for c in range(len(taus)):
            if not np.isnan(grid[r, c]):
                plt.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                         color="black", fontsize=8)
    plt.tight_layout()
    _save(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = p.parse_args()

    csv = os.path.join(args.results_dir, "noniid_sweep.csv")
    if not os.path.exists(csv):
        sys.exit(f"{csv} not found. Run scripts/sweep_noniid.py first.")
    df = pd.read_csv(csv)
    os.makedirs(args.figures_dir, exist_ok=True)

    tau = args.tau if args.tau in df["tau"].unique() else sorted(df["tau"].unique())[0]
    plot_fpr_vs_alpha(df, os.path.join(args.figures_dir, "noniid_fpr_vs_alpha.png"), tau)
    plot_detection_vs_alpha(df, os.path.join(args.figures_dir, "noniid_detection_vs_alpha.png"), tau)
    plot_fpr_heatmap(df, os.path.join(args.figures_dir, "noniid_fpr_heatmap.png"))


if __name__ == "__main__":
    main()
