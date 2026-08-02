PY ?= ./.venv/bin/python

.PHONY: help install smoke run iid noniid poisoned all figures tables test clean

help:
	@echo "make install   - create venv and install dependencies"
	@echo "make smoke      - fast smoke run of all 4 scenarios (few rounds)"
	@echo "make run        - full IID baseline (30 rounds)"
	@echo "make iid        - IID baseline"
	@echo "make noniid     - non-IID (Dirichlet) baseline"
	@echo "make poisoned   - poisoned FedAvg (attacks, no defence)"
	@echo "make all        - run all 4 scenarios + figures + summary table"
	@echo "make figures    - build the four-scenario figures from results/"
	@echo "make tables     - build the summary table (CSV + LaTeX) from results/"
	@echo "make test       - run unit + smoke tests"
	@echo "make clean      - remove results/ and figures/"

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

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf results figures
