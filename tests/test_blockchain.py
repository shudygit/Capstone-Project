"""Tests for the Module 3 blockchain ledger: PoW, linkage, tamper + signatures."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedblock.blockchain import (Blockchain, KeyRegistry, Transaction, sha256_hex,
                                 verify_signature)


def _tx(cid, h):
    return Transaction(client_id=cid, round=0, weight_hash=h, num_samples=100,
                       signature_hex="ab", accepted=True)


def test_pow_meets_difficulty():
    chain = Blockchain(difficulty=2)
    block = chain.add_block([])
    assert block.hash.startswith("00")          # Proof-of-Work satisfied
    assert block.hash == block.compute_hash()   # hash is self-consistent


def test_chain_valid_after_appends():
    chain = Blockchain(difficulty=1)
    for i in range(4):
        chain.add_block([_tx(i, sha256_hex(str(i).encode()))])
    assert chain.is_valid()
    assert len(chain) == 5                       # genesis + 4


def test_tampering_breaks_validity():
    chain = Blockchain(difficulty=1)
    chain.add_block([_tx(0, sha256_hex(b"abc"))])
    chain.add_block([_tx(1, sha256_hex(b"def"))])
    # Change a recorded update without re-mining: the chain must detect it.
    chain.chain[1].transactions[0].weight_hash = sha256_hex(b"TAMPERED")
    assert not chain.is_valid()


def test_signature_roundtrip_and_rejection():
    reg = KeyRegistry(key_bits=2048)
    alice = reg.register(0)
    reg.register(1)
    msg = sha256_hex(b"alice-update").encode()
    sig = alice.sign(msg)
    assert verify_signature(reg.public_key(0), msg, sig)        # correct key accepts
    assert not verify_signature(reg.public_key(1), msg, sig)    # wrong key rejects
    assert not verify_signature(reg.public_key(0), b"other", sig)  # tampered msg rejects
