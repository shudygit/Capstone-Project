# Federated Learning: Baseline + Poisoning Attacks (FedAvg on MNIST)

Capstone thesis *Hybrid Blockchain-Assisted Poisoning Detection in Federated
Learning* - **Shudhatm Jain (25253301)**, School of Computer Science, University
of Galway. Supervisor: Dr. Malika Bendechache.

This repository contains the first two modules of the project:

- **Module 1 - FedAvg baseline:** simulated clients collaboratively train a small
  CNN on MNIST with **Federated Averaging**, no attacks, no defences. The clean
  reference point.
- **Module 2 - Poisoning attacks:** **label-flipping** and **gradient-noise
  injection** by malicious clients, used to measure how much the baseline
  degrades under adversarial conditions (baseline vulnerability). No defence yet
  - that is a later module.

## What these modules do

- Partition MNIST across **N simulated clients** under **IID** or **non-IID
  (Dirichlet alpha)** label skew.
- Each round: broadcast the global model -> clients run local SGD -> **FedAvg**
  aggregation (weighted by sample count) -> evaluate on the held-out test set.
- Optionally designate a fraction of clients as **malicious**, each running one
  of two attacks:
  - *label flipping* - trains on data whose `source` class is relabelled as
    `target` (stealthy: a valid-looking but wrong gradient);
  - *gradient-noise injection* - adds Gaussian noise (`noise_sigma`) to the
    client's final weights (statistically anomalous).
- Log per-round global accuracy, loss and training latency to CSV, and overlay
  baseline vs poisoned learning curves.

## Project layout

```
fedblock/                # the package
├── config.py            # typed YAML config (experiment/data/model/federated/attack)
├── data.py              # MNIST + IID / Dirichlet partitioning
├── models.py            # SmallCNN, MLP
├── attacks.py           # label flipping + gradient-noise injection (module 2)
├── client.py            # local SGD training (+ poisoning behaviour)
├── aggregator.py        # FedAvg
├── metrics.py           # test-set evaluation
└── server.py            # round orchestration + malicious-client assignment
configs/baseline.yaml    # clean baseline configuration
configs/poisoned.yaml    # attacks enabled (module 2)
scripts/                 # run_experiment.py, make_figures.py
tests/                   # unit (fedavg, attacks) + end-to-end smoke tests
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
handshake is ever performed. The project therefore runs fully offline. The
compressed `.gz` files are version-controlled and are extracted automatically on
first run.

## Usage

```bash
# Fast smoke run (3 rounds):
make smoke

# Full IID baseline (30 rounds):
./.venv/bin/python scripts/run_experiment.py --config configs/baseline.yaml

# Non-IID (Dirichlet, more skew):
./.venv/bin/python scripts/run_experiment.py --config configs/baseline.yaml \
    --set data.partition=dirichlet --set data.dirichlet_alpha=0.1

# Poisoned run (module 2): attacks active, no defence:
./.venv/bin/python scripts/run_experiment.py --config configs/poisoned.yaml

# Degradation study: baseline + poisoned + overlaid figures in one go:
make compare

# Plot the learning curves (overlays every condition found in results/):
./.venv/bin/python scripts/make_figures.py
```

Any config field can be overridden on the command line with `--set section.key=value`,
including the attack knobs - e.g. `--set attack.malicious_fraction=0.5`,
`--set attack.noise_sigma=1.0`, `--set "attack.types=[label_flip]"`.

Outputs land in `results/` (per-round CSV + summary JSON) and `figures/`.

## Tests

```bash
make test        # FedAvg + attacks unit tests + short end-to-end smoke run
```

## Roadmap (thesis modules)

1. **Baseline** - FedAvg on MNIST, IID + non-IID. ✅
2. **Poisoning attacks** - label flipping + gradient-noise injection. ✅
3. Blockchain ledger - SHA-256 hashes, RSA-2048 signatures, Proof-of-Work.
4. Z-score Byzantine filter + the four-scenario evaluation.

## AI-use disclosure

Consistent with the capstone AI-transparency guidelines, this codebase was
scaffolded with assistance from Claude (Anthropic). All experimental design,
research framing, analysis and conclusions are the author's own.
