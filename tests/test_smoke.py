"""End-to-end smoke test: a short baseline run must complete and improve.

Skips automatically if MNIST cannot be downloaded (offline CI).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.config import load_config
from fedblock.server import FederatedServer

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")


def test_baseline_runs_and_learns():
    cfg = load_config(CONFIG, overrides={
        "federated": {"num_rounds": 2, "device": "cpu"},
        "data": {"num_clients": 6},
    })
    try:
        server = FederatedServer(cfg)
    except Exception as e:  # pragma: no cover - network/data availability
        pytest.skip(f"Could not initialise (likely MNIST download): {e}")
    history = server.run(verbose=False)
    assert len(history) == 2
    assert 0.0 <= history[-1]["test_acc"] <= 1.0
    # Two rounds of FedAvg on MNIST should comfortably clear chance (10%).
    assert history[-1]["test_acc"] > 0.5
