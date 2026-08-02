"""Tests for the Module 4 z-score anomaly filter."""
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.defense import zscore_filter
from fedblock.metrics import detection_metrics


def _state(scale=1.0, shift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return OrderedDict([
        ("w", torch.randn(40, 40, generator=g) * scale + shift),
        ("b", torch.randn(40, generator=g) * scale + shift),
    ])


def test_flags_obvious_outlier():
    # Eight near-identical honest updates and one wildly different malicious one.
    updates = {i: _state(scale=0.01, seed=i) for i in range(8)}
    updates[8] = _state(scale=5.0, shift=10.0, seed=99)
    result = zscore_filter(updates, z_threshold=2.5, fraction_threshold=0.05)
    assert 8 in result.flagged
    assert all(c not in result.flagged for c in range(8))


def test_no_false_positive_on_similar_clients():
    updates = {i: _state(scale=0.01, seed=i) for i in range(8)}
    result = zscore_filter(updates, z_threshold=2.5, fraction_threshold=0.1)
    assert result.flagged == []


def test_too_few_clients_flags_nobody():
    updates = {0: _state(seed=0), 1: _state(scale=10.0, seed=1)}
    assert zscore_filter(updates).flagged == []


def test_detection_metrics_counts():
    # Participants 0..4; clients 1 and 3 are malicious; filter flagged 1 and 2.
    m = detection_metrics(flagged={1, 2}, malicious={1, 3}, participating={0, 1, 2, 3, 4})
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 2)
    assert m.detection_rate == 0.5          # caught 1 of 2 malicious
    assert abs(m.false_positive_rate - 1 / 3) < 1e-9  # 1 of 3 honest wrongly flagged
