PY ?= ./.venv/bin/python

.PHONY: help install smoke run iid noniid figures test clean

help:
	@echo "make install   - create venv and install dependencies"
	@echo "make smoke      - fast baseline smoke run (3 rounds)"
	@echo "make run        - full IID baseline (30 rounds)"
	@echo "make iid        - IID baseline"
	@echo "make noniid     - non-IID (Dirichlet) baseline"
	@echo "make figures    - plot accuracy/loss curves from results/"
	@echo "make test       - run unit + smoke tests"
	@echo "make clean      - remove results/ and figures/"

install:
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) scripts/run_experiment.py --quick

run: iid

iid:
	$(PY) scripts/run_experiment.py --config configs/baseline.yaml --set data.partition=iid

noniid:
	$(PY) scripts/run_experiment.py --config configs/baseline.yaml \
		--set data.partition=dirichlet --set data.dirichlet_alpha=0.1

figures:
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf results figures
