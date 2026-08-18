"""Tests for the published defences re-implemented for comparison (FedECPA, FLTrust)."""
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.baselines import fltrust_aggregate, fltrust_scores, iqr_filter


def _wide(seed=0):
    """A homogeneous update with a realistic number of parameters."""
    g = torch.Generator().manual_seed(seed)
    return OrderedDict([("w", torch.randn(200, 200, generator=g) * 0.01)])


def _state(scale=1.0, shift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return OrderedDict([
        ("w", torch.randn(40, 40, generator=g) * scale + shift),
        ("b", torch.randn(40, generator=g) * scale + shift),
    ])


# --- FedECPA (IQR filter) -------------------------------------------------
def test_iqr_flags_obvious_outlier():
    updates = {i: _state(scale=0.01, seed=i) for i in range(8)}
    updates[8] = _state(scale=5.0, shift=10.0, seed=99)
    assert iqr_filter(updates).flagged == [8]


def test_iqr_keeps_a_homogeneous_cohort():
    # Needs a realistic parameter count: the share of out-of-fence weights is a
    # noisy estimate on a few hundred dimensions but settles down on real models.
    updates = {i: _wide(seed=i) for i in range(10)}
    assert iqr_filter(updates).flagged == []


def test_iqr_is_noisier_than_the_zscore_filter_on_honest_clients():
    """IQR fences sit much closer to the tolerance than the z-score cutoff does.

    This is why the two cannot simply share a threshold without checking: on an
    honest cohort the z-score filter leaves a wide margin under tau, while the
    textbook 1.5*IQR rule uses most of it up.
    """
    from fedblock.defense import zscore_filter
    updates = {i: _wide(seed=i) for i in range(10)}
    z_frac = sum(zscore_filter(updates).fraction_extreme.values()) / 10
    q_frac = sum(iqr_filter(updates).fraction_extreme.values()) / 10
    assert z_frac < 0.01           # comfortable margin below tau = 0.05
    assert q_frac > z_frac * 5     # far closer to the tolerance


def test_iqr_is_a_no_op_on_tiny_cohorts():
    # Quantiles over two clients say nothing useful, so nobody is flagged.
    updates = {0: _state(seed=0), 1: _state(scale=9.0, seed=1)}
    assert iqr_filter(updates).flagged == []


# --- FLTrust --------------------------------------------------------------
def _zeros():
    return OrderedDict([("w", torch.zeros(10)), ("b", torch.zeros(4))])


def test_fltrust_scores_zero_for_opposing_direction():
    glob = _zeros()
    server = OrderedDict([("w", torch.ones(10)), ("b", torch.ones(4))])
    aligned = OrderedDict([("w", torch.ones(10) * 2), ("b", torch.ones(4) * 2)])
    opposed = OrderedDict([("w", -torch.ones(10)), ("b", -torch.ones(4))])
    scores = fltrust_scores({0: aligned, 1: opposed}, glob, server)
    assert scores[0] > 0.99          # same direction, trusted
    assert scores[1] == 0.0          # opposite direction, excluded


def test_fltrust_rescales_an_inflated_update():
    # A client scaled 100x should not move the model 100x further than the server.
    glob = _zeros()
    server = OrderedDict([("w", torch.ones(10)), ("b", torch.ones(4))])
    huge = OrderedDict([("w", torch.ones(10) * 100), ("b", torch.ones(4) * 100)])
    scores = fltrust_scores({0: huge}, glob, server)
    agg = fltrust_aggregate({0: huge}, glob, server, scores)
    # Magnitude is normalised back to the server's own update.
    assert torch.allclose(agg["w"], torch.ones(10), atol=1e-4)


def test_fltrust_holds_the_model_when_nobody_is_trusted():
    glob = OrderedDict([("w", torch.full((10,), 0.5)), ("b", torch.zeros(4))])
    server = OrderedDict([("w", torch.ones(10)), ("b", torch.ones(4))])
    opposed = OrderedDict([("w", torch.zeros(10)), ("b", -torch.ones(4))])
    scores = fltrust_scores({0: opposed}, glob, server)
    assert all(s == 0.0 for s in scores.values())
    agg = fltrust_aggregate({0: opposed}, glob, server, scores)
    assert torch.allclose(agg["w"], glob["w"])
