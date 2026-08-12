"""Tests for the temporal (ledger history) detector."""
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.blockchain import Blockchain, Transaction
from fedblock.temporal_detector import deviation_signals, flag_from_ledger


def _tx(cid, signal):
    return Transaction(client_id=cid, round=0, weight_hash="x", num_samples=100,
                       signature_hex="ab", accepted=True, deviation_signal=signal)


def _chain_with_history(per_round_signals):
    """Build a chain: per_round_signals is a list of {client_id: signal} per round."""
    chain = Blockchain(difficulty=1)
    for round_signals in per_round_signals:
        chain.add_block([_tx(c, s) for c, s in round_signals.items()])
    return chain


def test_flags_persistent_against_consensus_client():
    # Client 2 leans against the consensus (negative) every round; others scatter.
    history = []
    for _ in range(5):
        history.append({0: 0.1, 1: -0.1, 2: -5.0, 3: 0.05})
    chain = _chain_with_history(history)
    flagged = flag_from_ledger(chain, warmup=3, threshold=3.0)
    assert flagged == [2]


def test_no_flag_before_warmup():
    history = [{0: 0.1, 1: -0.1, 2: -5.0}]        # only one round of history
    chain = _chain_with_history(history)
    assert flag_from_ledger(chain, warmup=3, threshold=3.0) == []


def test_no_flag_when_all_similar():
    history = [{0: 0.1, 1: -0.1, 2: 0.05, 3: -0.05} for _ in range(5)]
    chain = _chain_with_history(history)
    assert flag_from_ledger(chain, warmup=3, threshold=3.0) == []


def test_deviation_signal_is_negative_for_opposer():
    # Global at 0; honest updates move consistently positive (consensus direction).
    g = torch.Generator().manual_seed(0)
    updates = {}
    for i in range(6):
        updates[i] = OrderedDict([("w", torch.ones(20) * 0.1 + torch.randn(20, generator=g) * 0.001)])
    # Attacker moves the opposite way (against the consensus).
    updates[6] = OrderedDict([("w", torch.ones(20) * -0.1)])
    global_state = OrderedDict([("w", torch.zeros(20))])
    signals = deviation_signals(updates, global_state)
    assert signals[6] < 0                          # opposes consensus
    assert signals[6] < min(signals[i] for i in range(6))  # most against
