PY ?= ./.venv/bin/python

.PHONY: help install smoke run iid noniid poisoned all figures tables \
        sweep-noniid sweep-adaptive robustness test clean

help:
	@echo "make install         - create venv and install dependencies"
	@echo "make smoke            - fast smoke run of all 4 scenarios (few rounds)"
	@echo "make run              - full IID baseline (30 rounds)"
	@echo "make poisoned         - poisoned FedAvg (attacks, no defence)"
	@echo "make all              - run all 4 scenarios + figures + summary table"
	@echo "make sweep-noniid     - non-IID false-positive sweep + figures"
	@echo "make sweep-adaptive   - adaptive-attacker frontier sweep + figure"
	@echo "make robustness       - both novelty sweeps + their figures"
	@echo "make test             - run all tests"
	@echo "make clean            - remove results/ and figures/"

install:
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) scripts/run_all.py --quick

run: iid

iid:
	$(PY) scripts/run_experiment.py --config configs/baseline.yaml --set data.partition=iid

noniid:
	$(PY) scripts/run_experiment.py --config configs/baseline.yaml \
		--set data.partition=dirichlet --set data.dirichlet_alpha=0.1

poisoned:
	$(PY) scripts/run_experiment.py --config configs/poisoned.yaml

all:
	$(PY) scripts/run_all.py
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_tables.py

figures:
	$(PY) scripts/make_figures.py

tables:
	$(PY) scripts/make_tables.py

sweep-noniid:
	$(PY) scripts/sweep_noniid.py
	$(PY) scripts/make_noniid_figures.py

sweep-adaptive:
	$(PY) scripts/sweep_adaptive.py
	$(PY) scripts/make_adaptive_figures.py

sweep-temporal:
	$(PY) scripts/sweep_temporal.py
	$(PY) scripts/make_temporal_figures.py

robustness: sweep-noniid sweep-adaptive sweep-temporal

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf results figures
