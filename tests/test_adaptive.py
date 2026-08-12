"""Tests for the adaptive (filter-aware) attacker."""
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.adaptive_attack import compute_reference, craft_adaptive_update
from fedblock.defense import zscore_filter


def _honest_update(seed):
    g = torch.Generator().manual_seed(seed)
    # Honest clients: near-identical small updates (low cross-client variance).
    return OrderedDict([
        ("w", torch.randn(30, 30, generator=g) * 0.01),
        ("b", torch.randn(30, generator=g) * 0.01),
    ])


def _global_state():
    return OrderedDict([("w", torch.zeros(30, 30)), ("b", torch.zeros(30))])


def test_crafting_hits_the_target_zscore():
    # Relative to the honest reference, every crafted weight must sit exactly
    # scale * z_threshold standard deviations from the group mean - the guarantee.
    updates = {i: _honest_update(i) for i in range(6)}
    ref = compute_reference(updates, honest_ids=list(range(6)))
    scale, z = 0.9, 2.5
    crafted = craft_adaptive_update(ref, _global_state(), z_threshold=z, scale=scale)
    for k, (mu, sigma) in ref.items():
        z_rel = (crafted[k] - mu) / sigma
        assert torch.allclose(z_rel.abs(), torch.full_like(z_rel, scale * z), atol=1e-4)


def test_crafted_update_evades_the_filter():
    # Eight honest clients form the reference; one adaptive attacker crafts at 0.9.
    updates = {i: _honest_update(i) for i in range(8)}
    ref = compute_reference(updates, honest_ids=list(range(8)))
    updates[8] = craft_adaptive_update(ref, _global_state(), z_threshold=2.5, scale=0.9)
    result = zscore_filter(updates, z_threshold=2.5, fraction_threshold=0.05)
    assert 8 not in result.flagged                       # attacker is NOT caught
    assert result.fraction_extreme[8] == 0.0             # zero coordinates are outliers


def test_more_aggressive_is_more_detectable():
    # A more aggressive attacker exposes more outlier weights (the evasion/damage
    # trade-off), even though a lone attacker inflates the group std and stays hard
    # to flag outright.
    def frac(scale):
        updates = {i: _honest_update(i) for i in range(8)}
        ref = compute_reference(updates, honest_ids=list(range(8)))
        updates[8] = craft_adaptive_update(ref, _global_state(), 2.5, scale)
        return zscore_filter(updates, z_threshold=2.5, fraction_threshold=0.05).fraction_extreme[8]

    assert frac(6.0) > frac(0.5)


def test_craft_is_deterministic():
    updates = {i: _honest_update(i) for i in range(4)}
    ref = compute_reference(updates, honest_ids=list(range(4)))
    a = craft_adaptive_update(ref, _global_state(), 2.5, 0.9)
    b = craft_adaptive_update(ref, _global_state(), 2.5, 0.9)
    assert torch.allclose(a["w"], b["w"])
