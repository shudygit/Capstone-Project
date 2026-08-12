"""Typed, YAML-backed configuration for the FedAvg baseline."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    seed: int = 0
    results_dir: str = "./results"


@dataclass
class DataConfig:
    dataset: str = "mnist"
    data_root: str = "./data"
    num_clients: int = 10
    partition: str = "iid"          # "iid" | "dirichlet"
    dirichlet_alpha: float = 0.5     # smaller alpha => more non-IID
    batch_size: int = 64
    test_batch_size: int = 1000


@dataclass
class ModelConfig:
    name: str = "small_cnn"          # "small_cnn" | "mlp"


@dataclass
class FederatedConfig:
    num_rounds: int = 30
    local_epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    client_fraction: float = 1.0     # fraction of clients sampled per round
    device: str = "auto"


"""Module 2: poisoning attacks against the FedAvg baseline."""

@dataclass
class AttackConfig:
    enabled: bool = False
    # malicious clients are split across the enabled attack types
    types: List[str] = field(default_factory=lambda: ["label_flip", "gradient_noise"])
    malicious_fraction: float = 0.3
    # label flipping
    label_flip_source: int = 7
    label_flip_target: int = 1
    label_flip_all: bool = False     # if True, map every label l -> (9 - l)
    # gradient / weight noise injection
    noise_sigma: float = 0.8         # std of Gaussian weight noise (clearly anomalous)
    noise_scale: float = 1.0         # multiplicative scaling of the update
    # adaptive (filter-aware) attacker - novelty extension
    adaptive_scale: float = 0.9      # crafts updates at |z| = scale * z_threshold (< 1 => evades)
    adaptive_mode: str = "whitebox"  # "whitebox" (current-round stats) | "graybox" (previous round)


@dataclass
class BlockchainConfig:
    """Module 3: blockchain ledger that records and audits every client update."""
    enabled: bool = False
    difficulty: int = 3              # Proof-of-Work: required leading zeros in the hash
    rsa_key_bits: int = 2048         # size of each client's signing key


@dataclass
class DefenseConfig:
    """Module 4: z-score anomaly filter that drops outlier updates before averaging."""
    enabled: bool = False
    z_threshold: float = 2.5         # a weight is 'extreme' if |z-score| exceeds this (epsilon)
    fraction_threshold: float = 0.05  # flag a client if this fraction of its weights are extreme (tau)
    # Temporal (history-based) detector - reads the blockchain ledger to catch a
    # persistent adaptive attacker that the per-round filter above misses.
    temporal: bool = False
    temporal_warmup: int = 3         # rounds of history to accumulate before flagging
    temporal_threshold: float = 3.0  # robust std-devs below the group to flag a client


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)

    _SECTIONS = {
        "experiment": ExperimentConfig,
        "data": DataConfig,
        "model": ModelConfig,
        "federated": FederatedConfig,
        "attack": AttackConfig,
        "blockchain": BlockchainConfig,
        "defense": DefenseConfig,
    }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Build a Config from a plain dict (parsed from YAML), section by section."""
        sections = {}
        for key, section_class in cls._SECTIONS.items():
            values = d.get(key, {}) or {}          # this section's dict, or empty
            # Catch typos: any key the dataclass does not define is rejected loudly.
            allowed = set(section_class.__dataclass_fields__)
            unknown = set(values) - allowed
            if unknown:
                raise ValueError(f"Unknown keys in '{key}': {sorted(unknown)}")
            sections[key] = section_class(**values)  # fill the dataclass
        return cls(**sections)

    def to_dict(self) -> Dict[str, Any]:
        """The reverse of from_dict: turn this Config back into a plain dict."""
        return {key: asdict(getattr(self, key)) for key in self._SECTIONS}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config, optionally layered with CLI overrides."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Config.from_dict(data)
