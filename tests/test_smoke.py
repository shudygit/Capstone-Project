"""End-to-end smoke tests: short runs of the baseline and the full hybrid.

Skips automatically if the local MNIST data cannot be loaded.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.config import load_config
from fedblock.server import FederatedServer

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


def _run(scenario_file, overrides):
    cfg = load_config(os.path.join(CONFIGS, scenario_file), overrides=overrides)
    try:
        server = FederatedServer(cfg)
    except Exception as e:  # pragma: no cover - data availability
        pytest.skip(f"Could not initialise (likely missing MNIST data): {e}")
    return server, server.run(verbose=False)


def test_baseline_runs_and_learns():
    _, history = _run("baseline.yaml", {
        "federated": {"num_rounds": 2, "device": "cpu"},
        "data": {"num_clients": 6},
    })
    assert len(history) == 2
    # Two rounds of FedAvg on MNIST should comfortably clear chance (10%).
    assert history[-1]["test_acc"] > 0.5


def test_full_hybrid_runs_and_chain_is_valid():
    server, history = _run("full_hybrid.yaml", {
        "federated": {"num_rounds": 2, "device": "cpu"},
        "data": {"num_clients": 10},
        "blockchain": {"difficulty": 1},
    })
    assert len(history) == 2
    assert 0.0 <= history[-1]["test_acc"] <= 1.0
    # The blockchain must be intact and the filter must never flag every client.
    assert server.chain.is_valid()
    assert history[-1]["num_aggregated"] >= 1
