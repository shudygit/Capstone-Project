#!/usr/bin/env python
"""Build the four-scenario summary table (CSV + LaTeX) for the thesis.

Per scenario (averaged over seeds) it reports:
  * final global accuracy
  * mean detection rate and false-positive rate (of the z-score filter)
  * attack success reduction - how much of the accuracy lost to poisoning is
    recovered, relative to the baseline-vs-poisoned gap:
        ASR = (acc[scenario] - acc[poisoned]) / (acc[baseline] - acc[poisoned])
  * mean per-round blockchain and filter overhead
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

SCENARIO_ORDER = ["baseline", "poisoned_nodefense", "blockchain_only", "full_hybrid"]
PRETTY = {
    "baseline": "Baseline",
    "poisoned_nodefense": "Poisoned, no defence",
    "blockchain_only": "Blockchain only",
    "full_hybrid": "Full hybrid",
}


def load_all(results_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(results_dir, "*seed*.csv"))
    if not files:
        sys.exit(f"No result CSVs in {results_dir}. Run scripts/run_all.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def final_acc_by_scenario(df: pd.DataFrame) -> pd.Series:
    """Mean over seeds of each scenario's last-round accuracy."""
    last = df.loc[df.groupby(["scenario", "seed"])["round"].idxmax()]
    return last.groupby("scenario")["test_acc"].mean()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    final = final_acc_by_scenario(df)
    base = final.get("baseline", float("nan"))
    pois = final.get("poisoned_nodefense", float("nan"))
    gap = base - pois

    rows = []
    for sc in SCENARIO_ORDER:
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue
        acc = final.get(sc, float("nan"))
        if sc in ("blockchain_only", "full_hybrid") and gap and gap == gap:
            asr = (acc - pois) / gap          # fraction of the lost accuracy recovered
        else:
            asr = float("nan")
        rows.append({
            "scenario": PRETTY[sc],
            "final_acc": round(acc, 4),
            "attack_success_reduction": round(asr, 4) if asr == asr else "--",
            "mean_detection_rate": round(sub["detection_rate"].mean(), 4),
            "mean_fpr": round(sub["false_positive_rate"].mean(), 4),
            "mean_blockchain_time_s": round(sub["blockchain_time"].mean(), 4),
            "mean_filter_time_s": round(sub["defense_time"].mean(), 5),
        })
    return pd.DataFrame(rows)


def to_latex(summary: pd.DataFrame) -> str:
    cols = ["scenario", "final_acc", "attack_success_reduction",
            "mean_detection_rate", "mean_fpr", "mean_blockchain_time_s"]
    head = ["Scenario", "Final Acc.", "ASR", "Det. Rate", "FPR", "Chain (s)"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Four-scenario results on MNIST (mean over seeds).}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}", r"\hline",
        " & ".join(head) + r" \\", r"\hline",
    ]
    for _, r in summary.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--out-dir", default="./results")
    args = p.parse_args()

    df = load_all(args.results_dir)
    summary = build_summary(df)
    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "summary_table.csv"), index=False)
    with open(os.path.join(args.out_dir, "summary_table.tex"), "w") as f:
        f.write(to_latex(summary))

    print(summary.to_string(index=False))
    print(f"\nSaved {os.path.join(args.out_dir, 'summary_table.csv')}")
    print(f"Saved {os.path.join(args.out_dir, 'summary_table.tex')}")


if __name__ == "__main__":
    main()
