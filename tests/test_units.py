"""Unit tests for the baseline building blocks (no data download needed)."""
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.aggregator import fedavg
from fedblock.config import load_config
from fedblock.models import build_model


def test_fedavg_weighted_average():
    a = OrderedDict([("w", torch.zeros(3))])
    b = OrderedDict([("w", torch.ones(3) * 3.0)])
    # 1 sample of zeros, 3 samples of threes -> weighted mean = 2.25
    out = fedavg([(a, 1), (b, 3)])
    assert torch.allclose(out["w"], torch.full((3,), 2.25))


def test_fedavg_rejects_empty():
    try:
        fedavg([])
    except ValueError:
        return
    raise AssertionError("fedavg should raise on empty input")


def test_build_model_shapes():
    for name in ("small_cnn", "mlp"):
        model = build_model(name)
        out = model(torch.randn(4, 1, 28, 28))
        assert out.shape == (4, 10)


def test_config_override():
    here = os.path.dirname(__file__)
    cfg = load_config(os.path.join(here, "..", "configs", "baseline.yaml"),
                      overrides={"data": {"partition": "dirichlet"}, "federated": {"num_rounds": 5}})
    assert cfg.data.partition == "dirichlet"
    assert cfg.federated.num_rounds == 5
