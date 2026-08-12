"""Temporal (history-based) detector that reads the blockchain ledger.

Why this exists
---------------
The per-round z-score filter is stateless: it only looks at one round at a time.
An adaptive attacker exploits this by staying just inside the threshold every
round, so it is never flagged. But that attacker has to lean against the honest
majority in the SAME direction round after round, and the blockchain has recorded
every one of those rounds in a tamper-proof ledger. So even though no single round
looks suspicious, the attacker's *history* does.

The idea
--------
1. Each round we measure how much every client leaned against the round's
   consensus (a single number per client, the "deviation signal"). Honest clients
   scatter around zero; a persistent attacker sits consistently on one side.
2. That number is stored on-chain in the client's transaction.
3. The detector adds up each client's signal across all rounds recorded on the
   ledger (a running total, like a CUSUM). Over time the attacker's total drifts
   far from the others.
4. After a short warm-up, we flag any client whose running total is a strong
   low-side outlier (using a robust median/MAD test, so a few attackers cannot
   hide the threshold).

This is deliberately simple. It is not a new idea in general (history- and
reputation-based detection exist, e.g. FoolsGold and on-chain temporal trust);
the point here is that the blockchain's immutable history is exactly the data such
a detector needs, and it catches the adaptive attacker that the per-round filter
misses.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch

from .utils import state_to_vector


def deviation_signals(updates: Dict[int, Dict[str, torch.Tensor]],
                      global_state: Dict[str, torch.Tensor]) -> Dict[int, float]:
    """One number per client: how far it leaned against the round's consensus.

    The consensus direction is (cohort mean update - global model). A client's
    deviation from the mean is projected onto that direction. Honest clients give
    values near zero; an attacker that opposes the consensus gives a consistently
    negative value.
    """
    client_ids = sorted(updates)
    vectors = {c: state_to_vector(updates[c]) for c in client_ids}

    # Use the coordinate-wise MEDIAN as the consensus, not the mean. The median is
    # robust to a minority of attackers, so even strong attackers cannot drag the
    # reference toward themselves and hide.
    stacked = torch.stack([vectors[c] for c in client_ids])
    center = stacked.median(dim=0).values
    consensus = center - state_to_vector(global_state)
    denom = consensus.norm() + 1e-8

    signals: Dict[int, float] = {}
    for c in client_ids:
        deviation = vectors[c] - center
        # Positive = leans with the consensus, negative = leans against it.
        signals[c] = float(torch.dot(deviation, consensus) / denom)
    return signals


def flag_from_ledger(chain, warmup: int = 3, threshold: float = 3.0) -> List[int]:
    """Read every client's signal history off the chain and flag persistent outliers.

    Args:
        chain: the Blockchain object (its blocks hold the per-round transactions).
        warmup: do not flag anyone until this many rounds have been recorded.
        threshold: how many robust standard deviations below the group a client's
            running total must be to get flagged.
    """
    running_total: Dict[int, float] = defaultdict(float)
    rounds_seen: Dict[int, int] = defaultdict(int)

    # Walk the immutable ledger and add up each client's signal over all rounds.
    for block in chain.chain:
        for tx in block.transactions:
            running_total[tx.client_id] += tx.deviation_signal
            rounds_seen[tx.client_id] += 1

    max_rounds = max(rounds_seen.values(), default=0)
    active = sorted(running_total)
    if max_rounds < warmup or len(active) < 3:
        return []

    totals = torch.tensor([running_total[c] for c in active])
    median = totals.median()
    mad = (totals - median).abs().median() * 1.4826 + 1e-8  # robust standard deviation

    flagged: List[int] = []
    for i, c in enumerate(active):
        robust_z = (running_total[c] - median) / mad
        # Attackers drift to the negative (against-consensus) side.
        if robust_z < -threshold:
            flagged.append(c)
    return flagged
