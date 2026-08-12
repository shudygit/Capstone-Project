# Hybrid Blockchain-Assisted Poisoning Detection in Federated Learning

Capstone thesis - **Shudhatm Jain (25253301)**, School of Computer Science,
University of Galway. Supervisor: Dr. Malika Bendechache.

This repository implements a **hybrid defence** for Federated Averaging (FedAvg) on
MNIST: a lightweight **blockchain ledger** plus a **z-score anomaly filter** that
together resist poisoning attacks. It is evaluated with a controlled
**four-scenario** design that isolates the contribution of each component.

## The four modules

- **Module 1 - FedAvg baseline.** Simulated clients collaboratively train a small
  CNN on MNIST, no attacks, no defences. The clean reference point.
- **Module 2 - Poisoning attacks.** `label-flipping` and `gradient-noise injection`
  by malicious clients, to measure how much the baseline degrades.
- **Module 3 - Blockchain ledger.** Each update is SHA-256 hashed, RSA-2048 signed
  by its client, and recorded in a Proof-of-Work, hash-linked chain. This provides
  a tamper-evident audit trail and an identity check. It audits, it does not filter.
- **Module 4 - Z-score anomaly filter.** Before averaging, clients whose weights are
  statistical outliers versus the group are detected and dropped.

## The four scenarios

| # | Scenario | Attack | Blockchain | Filter | What it isolates |
|---|----------|:------:|:----------:|:------:|------------------|
| i   | `baseline`            | off | off | off | Clean FedAvg reference |
| ii  | `poisoned_nodefense`  | on  | off | off | Raw attack damage |
| iii | `blockchain_only`     | on  | on  | off | What the ledger alone adds (ii vs iii) |
| iv  | `full_hybrid`         | on  | on  | on  | What the filter adds on top (iii vs iv) |

**Representative result** (MNIST, 10 clients, 30 % malicious): baseline ~0.97,
poisoned ~0.67, blockchain-only ~0.67 (identical accuracy, plus an audit trail and
overhead), full hybrid ~0.95. The filter catches the statistically loud
noise attack (detection rate for that attacker, zero false positives) while the
stealthy label-flip attack remains hard to spot, which is an informative finding.

## Project layout

```
fedblock/                # the package
├── config.py            # typed YAML config (experiment/data/model/federated/attack/blockchain/defense)
├── data.py              # offline MNIST + IID / Dirichlet partitioning
├── models.py            # SmallCNN, MLP
├── attacks.py           # label flipping + gradient-noise injection (module 2)
├── client.py            # local SGD training (+ poisoning behaviour)
├── aggregator.py        # FedAvg
├── blockchain.py        # SHA-256 + RSA-2048 + Proof-of-Work ledger (module 3)
├── defense.py           # z-score anomaly filter (module 4)
├── metrics.py           # test-set evaluation + detection metrics
└── server.py            # round orchestration + four-scenario logic
configs/                 # one YAML per scenario (baseline, poisoned, blockchain_only, full_hybrid)
scripts/                 # run_experiment, run_all, make_figures, make_tables
tests/                   # unit (fedavg, attacks, blockchain, defense) + end-to-end smoke
```

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
#  or:  make install
```

> PyTorch selects the best device automatically (CUDA -> Apple MPS -> CPU).
> Python 3.12 is recommended for stable PyTorch wheels.

**The dataset ships with this repository.** MNIST lives in `data/MNIST/raw` and is
loaded strictly from local files: no download, no network access and no SSL
handshake is ever performed. The project runs fully offline. The compressed `.gz`
files are version-controlled and are extracted automatically on first run.

## Usage

```bash
# Fast smoke run of all four scenarios (few rounds):
make smoke

# Run one scenario at full length:
./.venv/bin/python scripts/run_experiment.py --config configs/full_hybrid.yaml

# Run all four scenarios, then build figures and the summary table:
make all
#   or across several seeds:
./.venv/bin/python scripts/run_all.py --seeds 0 1 2
./.venv/bin/python scripts/make_figures.py
./.venv/bin/python scripts/make_tables.py
```

Any config field can be overridden on the command line with `--set section.key=value`,
for example `--set attack.malicious_fraction=0.5`, `--set defense.z_threshold=2.0`,
`--set blockchain.difficulty=4`, `--set data.partition=dirichlet`.

Outputs land in `results/` (per-round CSV + summary JSON, plus `summary_table.csv`
and `summary_table.tex`) and `figures/` (accuracy curves, detection metrics, and the
per-round overhead breakdown).

## Metrics recorded

Per round: global accuracy and loss, number of clients flagged, detection rate and
false-positive rate of the filter, and a timing breakdown (training, blockchain,
filter). Across scenarios: attack success reduction (how much of the lost accuracy
the hybrid recovers).

## Robustness analysis (novelty extensions)

Two studies probe the limits of the z-score filter. These are the parts of the
work with the clearest research contribution, and both were named as future work
in the literature review.

**A. Non-IID false positives.** As clients become more heterogeneous, honest
updates diverge, so the filter starts flagging honest clients. `sweep_noniid.py`
sweeps the Dirichlet `alpha` against the filter `tau` and measures the
false-positive rate (on an honest-only cohort, to isolate the pure effect) and the
detection rate under attack.

```bash
python scripts/sweep_noniid.py --alphas 5.0 1.0 0.3 0.1 0.05 --taus 0.02 0.05 0.1
python scripts/make_noniid_figures.py
```

**B. Adaptive (filter-aware) attacker.** An attacker that knows the filter crafts
updates bounded to `|z| = adaptive_scale x z_threshold`. With `adaptive_scale < 1`
it is provably never flagged, yet it still biases the model. Two threat models are
implemented: `whitebox` (uses the current round's honest statistics) and `graybox`
(uses the previous round's). `sweep_adaptive.py` traces the evasion-vs-damage frontier.

```bash
python scripts/sweep_adaptive.py --scales 0.5 0.7 0.9 1.0 1.5 2.0 --modes whitebox graybox
python scripts/make_adaptive_figures.py
```

Key finding: a lone adaptive attacker inflates the very standard deviation the
filter uses to judge it, so it evades detection across a wide range of
aggressiveness; its damage is nonetheless bounded because it must stay near the
honest cohort. This motivates the temporal detector below.

**C. Temporal ledger-anchored detector (depth component).** The per-round filter
is stateless, so the adaptive attacker hides inside every single round. But it must
lean against the honest majority in the same direction every round, and the
blockchain has recorded all of them. The temporal detector (`fedblock/temporal_detector.py`)
stores a one-number "against-the-consensus" signal per client on-chain, then adds
each client's signal across the whole immutable ledger history (a CUSUM) and flags
persistent low-side outliers with a robust median/MAD test. Bans are sticky, and
the consensus is measured with the coordinate-wise median so strong attackers
cannot corrupt the reference.

```bash
python scripts/sweep_temporal.py --scales 0.9 1.5 2.0 3.0
python scripts/make_temporal_figures.py
```

Result: against the same adaptive attacker the per-round filter's accuracy falls
from 0.97 to 0.23 as the attack strengthens (detection 0.00 throughout), while the
temporal detector holds accuracy near 0.98 at every strength (detection rising to
1.0 after a short warm-up, zero false positives). This turns the earlier negative
result - a passive ledger adds no robustness - into a positive one: the ledger's
immutable *history* is exactly what a defence needs to catch an adaptive attacker.
This is an incremental combination, not a new idea (cf. FoolsGold, on-chain
temporal reputation); its value is the specific pairing and the clean result.

## Tests

```bash
make test        # 27 tests: fedavg, attacks, blockchain, filter, adaptive, temporal, smoke
```

## Roadmap (thesis modules)

1. **Baseline** - FedAvg on MNIST, IID + non-IID. Done.
2. **Poisoning attacks** - label flipping + gradient-noise injection. Done.
3. **Blockchain ledger** - SHA-256 hashes, RSA-2048 signatures, Proof-of-Work. Done.
4. **Z-score anomaly filter** + the four-scenario evaluation. Done.
5. **Robustness analysis** - non-IID false positives + adaptive (filter-aware) attacker. Done.
6. **Temporal ledger-anchored detector** - history-based detection that catches the adaptive attacker. Done.

## AI-use disclosure

Consistent with the capstone AI-transparency guidelines, this codebase was
scaffolded with assistance from Claude (Anthropic). All experimental design,
research framing, analysis and conclusions are the author's own.
