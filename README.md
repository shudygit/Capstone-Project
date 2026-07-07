# Federated Learning Baseline (FedAvg on MNIST)

**Module 1** of the capstone thesis *Hybrid Blockchain-Assisted Poisoning
Detection in Federated Learning* - **Shudhatm Jain (25253301)**, School of
Computer Science, University of Galway. Supervisor: Dr. Malika Bendechache.

This repository is the clean **federated learning baseline**: simulated clients
collaboratively train a small CNN on MNIST using **Federated Averaging (FedAvg)**,
with no attacks and no defences. It is the reference point against which the
later thesis modules — poisoning attacks, a blockchain audit ledger, and a
z-score Byzantine filter — are evaluated.

## What this module does

- Partitions MNIST across **N simulated clients** under **IID** or **non-IID
  (Dirichlet α)** label skew.
- Each round: broadcast the global model → clients run local SGD → **FedAvg**
  aggregation (weighted by sample count) → evaluate on the held-out test set.
- Logs per-round global accuracy, loss and training latency to CSV, and plots
  the learning curves.

## Project layout

```
fedblock/                # the package
├── config.py            # typed YAML config (experiment/data/model/federated)
├── data.py              # MNIST + IID / Dirichlet partitioning
├── models.py            # SmallCNN, MLP
├── client.py            # local SGD training
├── aggregator.py        # FedAvg
├── metrics.py           # test-set evaluation
└── server.py            # round orchestration
configs/baseline.yaml    # baseline configuration
scripts/                 # run_experiment.py, make_figures.py
tests/                   # unit + end-to-end smoke tests
```

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
#  or:  make install
```

> PyTorch selects the best device automatically (CUDA → Apple MPS → CPU). MNIST
> downloads on first run into `./data`. Python 3.12 is recommended for stable
> PyTorch wheels.

## Usage

```bash
# Fast smoke run (3 rounds):
make smoke

# Full IID baseline (30 rounds):
./.venv/bin/python scripts/run_experiment.py --config configs/baseline.yaml

# Non-IID (Dirichlet, more skew):
./.venv/bin/python scripts/run_experiment.py --config configs/baseline.yaml \
    --set data.partition=dirichlet --set data.dirichlet_alpha=0.1

# Plot the learning curves:
./.venv/bin/python scripts/make_figures.py
```

Any config field can be overridden on the command line with `--set section.key=value`
(e.g. `--set data.num_clients=5`, `--set federated.local_epochs=2`).

Outputs land in `results/` (per-round CSV + summary JSON) and `figures/`.

## Tests

```bash
make test        # FedAvg unit tests + short end-to-end smoke run
```

## Roadmap (later thesis modules)

1. **Baseline (this repo)** — FedAvg on MNIST, IID + non-IID. ✅
2. Poisoning attacks — label flipping + gradient-noise injection.
3. Blockchain ledger — SHA-256 hashes, RSA-2048 signatures, Proof-of-Work.
4. Z-score Byzantine filter + the four-scenario evaluation.

## AI-use disclosure

Consistent with the capstone AI-transparency guidelines, this codebase was
scaffolded with assistance from Claude (Anthropic). All experimental design,
research framing, analysis and conclusions are the author's own.
