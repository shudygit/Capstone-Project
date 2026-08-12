"""Adaptive (filter-aware) attacker - novelty extension.

The z-score filter flags a client when too many of its weights are statistical
outliers versus the group (|z-score| > z_threshold). An adaptive attacker that
KNOWS this rule can craft an update that stays under the threshold on every
coordinate, so it is provably never flagged, while still pushing the averaged
model in a harmful direction.

Threat models (both implemented, selected by attack.adaptive_mode):
  * whitebox - the attacker knows the CURRENT round's honest mean/std per weight.
               The strongest, worst-case 'full knowledge' assumption.
  * graybox  - the attacker only has the PREVIOUS round's honest statistics as an
               estimate. More realistic and weaker.

Crafting rule. Given the honest reference (mu, sigma) per weight, the global model
w_g, the filter threshold z, and a scale s in (0, 1):
    direction = -sign(mu - w_g)            # oppose the honest consensus
    crafted   = mu + direction * (s * z) * sigma
Every crafted weight sits exactly (s * z) standard deviations from the group mean,
so its |z-score| = s * z < z. With s < 1 the update cannot be flagged, yet it
biases every coordinate against the direction the honest clients agreed on.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch

# A per-weight honest reference: key -> (mean tensor, std tensor).
Reference = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def compute_reference(updates: Dict[int, Dict[str, torch.Tensor]],
                      honest_ids: List[int], eps: float = 1e-8) -> Reference:
    """Per-weight mean and std across the honest clients' updates.

    This is the attacker's estimate of the cohort statistics the filter will use.
    """
    ref: Reference = OrderedDict()
    keys = updates[honest_ids[0]].keys()
    for k in keys:
        stacked = torch.stack([updates[c][k].float() for c in honest_ids])  # [H, ...]
        mu = stacked.mean(dim=0)
        sigma = stacked.std(dim=0, unbiased=False).clamp(min=eps)
        ref[k] = (mu, sigma)
    return ref


def craft_adaptive_update(reference: Reference,
                          global_state: Dict[str, torch.Tensor],
                          z_threshold: float, scale: float) -> Dict[str, torch.Tensor]:
    """Build an evasive update bounded to |z| = scale * z_threshold on every weight.

    With scale < 1 the update is guaranteed to fall under the filter's threshold.
    """
    s = scale * z_threshold
    crafted: Dict[str, torch.Tensor] = OrderedDict()
    for k, (mu, sigma) in reference.items():
        # Direction that opposes how the honest clients moved from the global model.
        direction = -torch.sign(mu - global_state[k].float())
        direction[direction == 0] = 1.0            # break ties deterministically
        crafted[k] = mu + direction * s * sigma
    return crafted
