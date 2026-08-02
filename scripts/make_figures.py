#!/usr/bin/env python
"""Build the thesis figures from the per-round result CSVs.

Produces three PNG/PDF figures:
  * accuracy_curves   - test accuracy per round, all four scenarios overlaid
  * detection_metrics - detection rate and false-positive rate (full hybrid)
  * overhead          - mean per-round time split into training / blockchain / filter

Results from multiple seeds are averaged.
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

SCENARIO_ORDER = ["baseline", "poisoned_nodefense", "blockchain_only", "full_hybrid"]
LABELS = {
    "baseline": "(i) Baseline",
    "poisoned_nodefense": "(ii) Poisoned, no defence",
    "blockchain_only": "(iii) Blockchain only",
    "full_hybrid": "(iv) Full hybrid",
}
COLORS = {
    "baseline": "#2a9d8f",
    "poisoned_nodefense": "#e63946",
    "blockchain_only": "#f4a261",
    "full_hybrid": "#264653",
}


def load_all(results_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(results_dir, "*seed*.csv"))
    if not files:
        sys.exit(f"No result CSVs in {results_dir}. Run scripts/run_all.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _mean_curve(df: pd.DataFrame, scenario: str, col: str):
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        return None
    return sub.groupby("round")[col].mean()


def plot_accuracy(df: pd.DataFrame, out: str) -> None:
    plt.figure(figsize=(8, 5))
    for sc in SCENARIO_ORDER:
        curve = _mean_curve(df, sc, "test_acc")
        if curve is not None:
            plt.plot(curve.index, curve.values, label=LABELS[sc],
                     color=COLORS[sc], linewidth=2)
    plt.xlabel("Communication round")
    plt.ylabel("Global test accuracy")
    plt.title("Global accuracy across the four scenarios")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved {out}")


def plot_detection(df: pd.DataFrame, out: str) -> None:
    dr = _mean_curve(df, "full_hybrid", "detection_rate")
    fpr = _mean_curve(df, "full_hybrid", "false_positive_rate")
    if dr is None:
        print("No full_hybrid results; skipping detection figure.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(dr.index, dr.values, label="Detection rate (recall)",
             color="#2a9d8f", linewidth=2)
    plt.plot(fpr.index, fpr.values, label="False positive rate",
             color="#e63946", linewidth=2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Communication round")
    plt.ylabel("Rate")
    plt.title("Z-score filter detection performance (full hybrid)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved {out}")


def plot_overhead(df: pd.DataFrame, out: str) -> None:
    comp = (df.groupby("scenario")[["train_time", "blockchain_time", "defense_time"]]
              .mean().reindex(SCENARIO_ORDER).dropna(how="all"))
    labels = [LABELS[s] for s in comp.index]
    plt.figure(figsize=(8, 5))
    bottom = [0.0] * len(comp)
    for col, color, name in [("train_time", "#264653", "Training"),
                             ("blockchain_time", "#f4a261", "Blockchain"),
                             ("defense_time", "#2a9d8f", "Z-score filter")]:
        vals = comp[col].values
        plt.bar(labels, vals, bottom=bottom, label=name, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.ylabel("Mean time per round (s)")
    plt.title("Per-round time breakdown (overhead)")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    args = p.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    df = load_all(args.results_dir)
    plot_accuracy(df, os.path.join(args.figures_dir, "accuracy_curves.png"))
    plot_detection(df, os.path.join(args.figures_dir, "detection_metrics.png"))
    plot_overhead(df, os.path.join(args.figures_dir, "overhead.png"))


if __name__ == "__main__":
    main()
