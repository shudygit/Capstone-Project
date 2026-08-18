#!/usr/bin/env python
"""Print mean +/- std (over seeds) for the paper's tables, as LaTeX-ready rows.

Run this after a multi-seed run, e.g.:
    python scripts/run_all.py --rounds 20 --seeds 0 1 2
    python scripts/sweep_adaptive.py --scales 0.5 0.9 1.0 1.5 2.0 3.0 --seeds 0 1 2
    python scripts/sweep_temporal.py --scales 0.9 1.5 2.0 3.0 --seeds 0 1 2
    python scripts/sweep_noniid.py --seeds 0 1 2
    python scripts/paper_tables.py

It reads the CSVs in results/ and prints the numbers to paste into main.tex.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "..", "results")
SCEN = ["baseline", "poisoned_nodefense", "blockchain_only", "full_hybrid"]


def _pm(vals):
    vals = np.asarray(vals, float)
    return f"{vals.mean():.3f}$\\pm${vals.std():.3f}"


def _load(pattern):
    files = glob.glob(os.path.join(R, pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def four_scenarios():
    df = _load("*seed*.csv")
    if df is None or "scenario" not in df.columns:
        print("(no scenario CSVs found)")
        return
    # last-round accuracy per (scenario, seed)
    last = df.loc[df.groupby(["scenario", "seed"])["round"].idxmax()]
    print("\n=== Table: four scenarios (mean +/- std over seeds) ===")
    base = last[last.scenario == "baseline"].groupby("seed")["test_acc"].mean()
    pois = last[last.scenario == "poisoned_nodefense"].groupby("seed")["test_acc"].mean()
    for sc in SCEN:
        sub = df[df.scenario == sc]
        if sub.empty:
            continue
        acc = last[last.scenario == sc].groupby("seed")["test_acc"].mean()
        dr = sub.groupby("seed")["detection_rate"].mean()
        fpr = sub.groupby("seed")["false_positive_rate"].mean()
        chain = sub.groupby("seed")["blockchain_time"].mean()
        if sc in ("blockchain_only", "full_hybrid"):
            asr = ((acc.values - pois.values) / (base.values - pois.values))
            asr_s = _pm(asr)
        else:
            asr_s = "--"
        pretty = {"baseline": "Baseline", "poisoned_nodefense": "Poisoned",
                  "blockchain_only": "Blockchain only", "full_hybrid": "Full hybrid"}[sc]
        print(f"{pretty} & {_pm(acc)} & {asr_s} & {_pm(dr)} & {_pm(fpr)} & {_pm(chain)} \\\\")


def adaptive():
    df = _load("adaptive_sweep.csv")
    if df is None:
        print("\n(no adaptive_sweep.csv)")
        return
    print("\n=== Adaptive sweep: acc (DR) per mode/scale ===")
    for mode in ["whitebox", "graybox"]:
        for scale in sorted(df[df["mode"] == mode]["scale"].dropna().unique()):
            s = df[(df["mode"] == mode) & (df["scale"] == scale)]
            print(f"{mode} s={scale}: acc={_pm(s.final_acc)} DR={_pm(s.mean_detection_rate)}")


def temporal():
    df = _load("temporal_sweep.csv")
    if df is None:
        print("\n(no temporal_sweep.csv)")
        return
    print("\n=== Temporal sweep: per-round vs temporal ===")
    for scale in sorted(df["scale"].unique()):
        row = []
        for defc in ["per_round", "temporal"]:
            s = df[(df.defence == defc) & (df.scale == scale)]
            row.append(f"{_pm(s.final_acc)} ({_pm(s.mean_detection_rate)})")
        print(f"{scale} & {row[0]} & {row[1]} \\\\")


def noniid():
    df = _load("noniid_sweep.csv")
    if df is None:
        print("\n(no noniid_sweep.csv)")
        return
    print("\n=== Non-IID honest-only FPR by alpha (tau=0.05) ===")
    h = df[(df.condition == "no_attack") & (df.tau == 0.05)]
    for a in sorted(h.alpha.unique(), reverse=True):
        s = h[h.alpha == a]
        print(f"alpha={a}: FPR={_pm(s.mean_fpr)}")


def comparison():
    """Table: our z-score filter against FLTrust and FedECPA on the same setup."""
    df = _load("*seed*.csv")
    if df is None or "scenario" not in df.columns:
        print("\n(no scenario CSVs found)")
        return
    last = df.loc[df.groupby(["scenario", "seed"])["round"].idxmax()]
    base = last[last.scenario == "baseline"].groupby("seed")["test_acc"].mean()
    pois = last[last.scenario == "poisoned_nodefense"].groupby("seed")["test_acc"].mean()
    if base.empty or pois.empty:
        print("\n(need baseline + poisoned runs to compute ASR)")
        return

    print("\n=== Table: defence comparison, same 10-client MNIST setup ===")
    rows = [("full_hybrid", "This work (z-score)"),
            ("fltrust", "FLTrust [2]"),
            ("fedecpa", "FedECPA [10]")]
    for key, pretty in rows:
        sub = df[df.scenario == key]
        if sub.empty:
            print(f"{pretty}: (not run)")
            continue
        acc = last[last.scenario == key].groupby("seed")["test_acc"].mean()
        seeds = sorted(set(acc.index) & set(base.index) & set(pois.index))
        asr = ((acc[seeds].values - pois[seeds].values)
               / (base[seeds].values - pois[seeds].values))
        dr = sub.groupby("seed")["detection_rate"].mean()
        fpr = sub.groupby("seed")["false_positive_rate"].mean()
        dfe = sub.groupby("seed")["defense_time"].mean()
        print(f"{pretty} & {_pm(acc)} & {_pm(asr)} & {_pm(dr)} & {_pm(fpr)} "
              f"& {_pm(dfe)} \\\\")


if __name__ == "__main__":
    four_scenarios()
    comparison()
    adaptive()
    temporal()
    noniid()
