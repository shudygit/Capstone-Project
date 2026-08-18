"""Published defences re-implemented on this setup, for a like-for-like comparison.

The z-score filter in ``defense.py`` is positioned against two existing methods, so
both are run here on the same ten-client MNIST setup, the same attacks and the same
seeds. Without that, there is no way to say whether the filter is better, worse or
the same as what already exists.

Two methods are implemented:

* ``iqr_filter`` - the FedECPA approach. Structurally identical to our z-score
  filter but it replaces mean/std with the interquartile range, which is robust to
  the outliers it is trying to find. A weight is extreme if it falls outside
  [Q1 - k*IQR, Q3 + k*IQR] across clients, and a client is flagged when too large a
  share of its weights are extreme.

* ``fltrust_*`` - the FLTrust approach. The server keeps a small clean root dataset,
  trains on it to get its own reference update, and scores each client by the cosine
  similarity between that client's update direction and the server's. Negative
  similarity scores zero, so the client is dropped. Surviving updates are rescaled to
  the server update's magnitude before the trust-weighted average, which is what
  neutralises scaling attacks.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch

from .defense import FilterResult
from .utils import state_to_vector


# FedECPA: IQR-based outlier filter

def iqr_filter(updates: Dict[int, Dict[str, torch.Tensor]],
               k: float = 1.5,
               fraction_threshold: float = 0.05) -> FilterResult:
    """Flag clients with too many weights outside the group's IQR fences.

    Args:
        updates: mapping client_id -> state_dict.
        k: IQR multiplier for the fences (1.5 is the textbook Tukey rule).
        fraction_threshold: tolerated share of out-of-fence weights (same tau as
            the z-score filter, so the two are compared at matched tolerance).
    """
    client_ids = sorted(updates.keys())
    if len(client_ids) <= 2:
        return FilterResult([], {c: 0.0 for c in client_ids})

    vectors = torch.stack([state_to_vector(updates[c]) for c in client_ids])  # [N, D]
    q1 = vectors.quantile(0.25, dim=0)
    q3 = vectors.quantile(0.75, dim=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    extreme = ((vectors < lower) | (vectors > upper)).float()
    fraction = extreme.mean(dim=1)

    flagged: List[int] = []
    fraction_extreme: Dict[int, float] = {}
    for i, c in enumerate(client_ids):
        frac = float(fraction[i].item())
        fraction_extreme[c] = frac
        if frac > fraction_threshold:
            flagged.append(c)
    return FilterResult(flagged, fraction_extreme)


# FLTrust: cosine trust against a server-held clean root dataset
def _delta_vector(state: Dict[str, torch.Tensor],
                  global_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The update direction: this client's weights minus the broadcast weights."""
    return state_to_vector(state) - state_to_vector(global_state)


def fltrust_scores(updates: Dict[int, Dict[str, torch.Tensor]],
                   global_state: Dict[str, torch.Tensor],
                   server_state: Dict[str, torch.Tensor],
                   eps: float = 1e-12) -> Dict[int, float]:
    """Trust score per client: ReLU of the cosine similarity with the server update.

    A client pulling against the server's clean reference direction scores zero and
    is excluded entirely; the rest are trusted in proportion to their alignment.
    """
    g0 = _delta_vector(server_state, global_state)
    n0 = g0.norm().clamp(min=eps)
    scores: Dict[int, float] = {}
    for c in sorted(updates):
        gi = _delta_vector(updates[c], global_state)
        cos = float(torch.dot(gi, g0) / (gi.norm().clamp(min=eps) * n0))
        scores[c] = max(0.0, cos)
    return scores


def fltrust_aggregate(updates: Dict[int, Dict[str, torch.Tensor]],
                      global_state: Dict[str, torch.Tensor],
                      server_state: Dict[str, torch.Tensor],
                      scores: Dict[int, float],
                      eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    """Trust-weighted FedAvg with every update rescaled to the server's magnitude.

    Rescaling is what stops a scaled-up malicious update from dominating: however
    large a client makes its update, it is normalised back to the length of the
    server's own before it is weighted.
    """
    total = sum(scores.values())
    if total <= eps:
        # Nobody aligned with the server this round; keep the model as it is.
        return OrderedDict((k, v.detach().clone()) for k, v in global_state.items())

    g0_norm = _delta_vector(server_state, global_state).norm().clamp(min=eps)

    agg: Dict[str, torch.Tensor] = OrderedDict()
    for key in global_state:
        agg[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    for c in sorted(updates):
        ts = scores.get(c, 0.0)
        if ts <= 0.0:
            continue
        gi_norm = _delta_vector(updates[c], global_state).norm().clamp(min=eps)
        scale = float(g0_norm / gi_norm)          # normalise magnitude to the server's
        w = ts / total
        for key in global_state:
            delta = updates[c][key].float() - global_state[key].float()
            agg[key] += w * scale * delta

    return OrderedDict((k, global_state[k].float() + agg[k]) for k in global_state)


def train_root_model(global_state: Dict[str, torch.Tensor], model, fed_cfg,
                     root_loader, device, num_steps: int) -> Dict[str, torch.Tensor]:
    """Train the server's reference model on its root data for ``num_steps`` steps.

    FLTrust rescales every client update to the magnitude of the server's own, so
    the server has to take roughly as many optimiser steps as a client does. The
    root set is far smaller than a client shard, so we cycle over it until the step
    budget is met rather than running a fixed number of epochs; one epoch on a
    hundred examples would produce a reference update so small that normalising the
    clients down to it would stall training entirely.
    """
    import torch.nn.functional as F
    from .utils import clone_state

    model.load_state_dict(global_state)
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=fed_cfg.lr,
                                momentum=fed_cfg.momentum)
    steps = 0
    while steps < num_steps:
        for x, y in root_loader:
            if steps >= num_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
            steps += 1
    return clone_state(model.state_dict())
