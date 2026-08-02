"""Module 3 - A lightweight blockchain ledger for federated model updates.

Every round, each client update is recorded as a signed transaction; the server
groups the round's transactions into a block, mines it with Proof-of-Work, and
appends it to a hash-linked chain. This gives three properties:

  * Integrity   - a SHA-256 hash of the update is stored on-chain, so any later
                  tampering with a recorded update no longer matches its hash.
  * Authenticity- each update is signed with the client's RSA-2048 private key and
                  checked with its public key, so the server can confirm an update
                  really came from a registered client (identity validation).
  * Immutability- blocks are linked by hash, so altering any past block breaks
                  every block after it (``is_valid`` detects this).

The ledger is deliberately self-contained (no external chain or crypto service)
so its overhead can be measured cleanly and separately from the training cost.
Note: the ledger *audits* updates, it does not decide which are malicious - that
is the job of the Module 4 filter.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# --------------------------------------------------------------------------- #
#  Cryptographic identity (RSA-2048 signatures)
# --------------------------------------------------------------------------- #
class ClientKey:
    """An RSA key pair for one client: signs with the private half."""

    def __init__(self, key_bits: int = 2048):
        self._private = rsa.generate_private_key(public_exponent=65537, key_size=key_bits)
        self.public = self._private.public_key()

    def sign(self, message: bytes) -> bytes:
        """Produce a signature for ``message`` using this client's private key."""
        return self._private.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )


def verify_signature(public_key, message: bytes, signature: bytes) -> bool:
    """Return True if ``signature`` is a valid signature of ``message`` for the key."""
    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False


class KeyRegistry:
    """Hands out and remembers one RSA key pair per client id."""

    def __init__(self, key_bits: int = 2048):
        self.key_bits = key_bits
        self._keys: Dict[int, ClientKey] = {}

    def register(self, client_id: int) -> ClientKey:
        if client_id not in self._keys:
            self._keys[client_id] = ClientKey(self.key_bits)
        return self._keys[client_id]

    def public_key(self, client_id: int):
        return self._keys[client_id].public


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hash of ``data`` as a hex string."""
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
#  Ledger data structures
# --------------------------------------------------------------------------- #
@dataclass
class Transaction:
    """One client's update record for a given round."""
    client_id: int
    round: int
    weight_hash: str          # SHA-256 of the serialised update
    num_samples: int
    signature_hex: str        # the client's signature over weight_hash
    accepted: bool            # did the signature verify?

    def to_dict(self) -> Dict:
        return self.__dict__


@dataclass
class Block:
    index: int
    transactions: List[Transaction]
    prev_hash: str
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0
    hash: str = ""

    def _payload(self) -> str:
        """The exact text whose hash defines this block (used for hashing / PoW)."""
        return json.dumps({
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "transactions": [t.to_dict() for t in self.transactions],
        }, sort_keys=True)

    def compute_hash(self) -> str:
        return sha256_hex(self._payload().encode())


class Blockchain:
    """A chain of mined blocks, each linked to the previous by its hash."""

    def __init__(self, difficulty: int = 3):
        self.difficulty = difficulty
        self.chain: List[Block] = []
        # The genesis block gives the chain a fixed starting point.
        genesis = Block(index=0, transactions=[], prev_hash="0" * 64)
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def _proof_of_work(self, block: Block) -> str:
        """Try nonces until the block hash starts with ``difficulty`` zeros."""
        target = "0" * self.difficulty
        block.nonce = 0
        h = block.compute_hash()
        while not h.startswith(target):
            block.nonce += 1
            h = block.compute_hash()
        return h

    def add_block(self, transactions: List[Transaction]) -> Block:
        block = Block(index=len(self.chain), transactions=transactions,
                      prev_hash=self.last_block.hash)
        block.hash = self._proof_of_work(block)
        self.chain.append(block)
        return block

    def is_valid(self) -> bool:
        """Check every block's hash, its link to the previous block, and its PoW."""
        target = "0" * self.difficulty
        for i in range(1, len(self.chain)):
            cur, prev = self.chain[i], self.chain[i - 1]
            if cur.hash != cur.compute_hash():   # block was tampered with
                return False
            if cur.prev_hash != prev.hash:       # link is broken
                return False
            if not cur.hash.startswith(target):  # PoW not satisfied
                return False
        return True

    def __len__(self) -> int:
        return len(self.chain)


@dataclass
class LedgerRoundStats:
    """Timing breakdown for one round of ledger activity (overhead measurement)."""
    sign_time: float = 0.0
    verify_time: float = 0.0
    hash_time: float = 0.0
    mine_time: float = 0.0
    num_transactions: int = 0
    num_rejected: int = 0

    @property
    def total_time(self) -> float:
        return self.sign_time + self.verify_time + self.hash_time + self.mine_time
