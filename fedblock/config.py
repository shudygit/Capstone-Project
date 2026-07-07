"""Typed, YAML-backed configuration for the FedAvg baseline."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

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


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)

    _SECTIONS = {
        "experiment": ExperimentConfig,
        "data": DataConfig,
        "model": ModelConfig,
        "federated": FederatedConfig,
    }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        kwargs: Dict[str, Any] = {}
        for key, klass in cls._SECTIONS.items():
            section = d.get(key, {}) or {}
            unknown = set(section) - set(klass.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unknown keys in '{key}': {sorted(unknown)}")
            kwargs[key] = klass(**section)
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {k: asdict(getattr(self, k)) for k in self._SECTIONS}


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
