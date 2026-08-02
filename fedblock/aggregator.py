"""FedAvg aggregation over client updates."""
from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Tuple
import torch


def fedavg(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Weighted average of client state_dicts, weighted by sample count.
    Args:
        updates: list of (state_dict, num_samples) for the participating clients.
    Returns:
        The aggregated global state_dict.
    """
    if not updates:
        raise ValueError("fedavg received no updates to aggregate")

    total = sum(n for _, n in updates)
    keys = updates[0][0].keys()
    agg: Dict[str, torch.Tensor] = OrderedDict()
    for k in keys:
        stacked = torch.stack([state[k].float() * (n / total) for state, n in updates])
        agg[k] = stacked.sum(dim=0)
    return agg
