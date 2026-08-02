"""Module 4 - Z-score anomaly filter (the lightweight statistical defence).

For the set of client updates in a round, we flatten each update into a vector and
look at every weight coordinate across clients. For each coordinate we compute the
group mean (mu) and standard deviation (sigma). A client is flagged as suspicious
if too large a fraction of its weights are statistical outliers:

    flag(client) = True  if  ( fraction of weights with |(w - mu) / sigma| > z_threshold )
                             > fraction_threshold

where ``z_threshold`` is the per-weight cutoff (the paper's epsilon) and
``fraction_threshold`` is the tolerated share of outlier weights (the paper's tau).
Flagged clients are removed before FedAvg averaging.

This needs no clean reference dataset and no cryptography; it is a handful of
tensor operations, negligible next to model training. It reliably catches the
noise-injection attack (statistically loud) but not label flipping (statistically
stealthy), which is an expected and informative result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .utils import state_to_vector


@dataclass
class FilterResult:
    flagged: List[int]                    # client ids removed from aggregation
    fraction_extreme: Dict[int, float]    # per-client share of outlier weights


def zscore_filter(updates: Dict[int, Dict[str, torch.Tensor]],
                  z_threshold: float = 2.5,
                  fraction_threshold: float = 0.05,
                  eps: float = 1e-8) -> FilterResult:
    """Flag clients whose updates are statistical outliers versus the group.

    Args:
        updates: mapping client_id -> state_dict (the update to be aggregated).
        z_threshold: per-weight |z-score| cutoff (epsilon).
        fraction_threshold: tolerated fraction of outlier weights (tau).
        eps: small floor on sigma so constant weights never divide by zero.
    """
    client_ids = sorted(updates.keys())
    if len(client_ids) <= 2:
        # Mean/std across 2 or fewer clients is not meaningful; flag nobody.
        return FilterResult([], {c: 0.0 for c in client_ids})

    # Stack all updates into one matrix: rows = clients, columns = weights.
    vectors = torch.stack([state_to_vector(updates[c]) for c in client_ids])  # [N, D]
    mu = vectors.mean(dim=0)                                    # [D]
    sigma = vectors.std(dim=0, unbiased=False).clamp(min=eps)   # [D]

    z = (vectors - mu) / sigma                 # z-score of every weight, per client
    extreme = (z.abs() > z_threshold).float()  # 1 where a weight is an outlier
    fraction = extreme.mean(dim=1)             # per client: share of outlier weights

    flagged: List[int] = []
    fraction_extreme: Dict[int, float] = {}
    for i, c in enumerate(client_ids):
        frac = float(fraction[i].item())
        fraction_extreme[c] = frac
        if frac > fraction_threshold:
            flagged.append(c)

    return FilterResult(flagged, fraction_extreme)
