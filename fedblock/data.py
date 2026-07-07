"""MNIST loading and IID / non-IID (Dirichlet) client partitioning."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# MNIST channel statistics used by virtually all reference implementations.
_MNIST_MEAN, _MNIST_STD = 0.1307, 0.3081


def _transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((_MNIST_MEAN,), (_MNIST_STD,)),
    ])


def load_mnist(data_root: str) -> Tuple[Dataset, Dataset]:
    """Download (if needed) and return the MNIST train and test datasets."""
    tfm = _transform()
    train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    return train, test


def iid_partition(train: Dataset, num_clients: int, seed: int) -> List[List[int]]:
    """Shuffle and split indices into ``num_clients`` equal IID shards."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    return [shard.tolist() for shard in np.array_split(idx, num_clients)]


def dirichlet_partition(train: Dataset, num_clients: int, alpha: float,
                        seed: int) -> List[List[int]]:
    """Non-IID partition: each client's class proportions drawn from Dir(alpha).

    Smaller ``alpha`` => more skewed (less IID) client distributions. This is the
    standard label-distribution-skew benchmark used in FL robustness papers.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(train.targets)
    num_classes = int(targets.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(class_idx, cuts)):
            client_indices[client_id].extend(chunk.tolist())

    for shard in client_indices:
        rng.shuffle(shard)
    return client_indices


def partition_data(train: Dataset, num_clients: int, partition: str,
                   dirichlet_alpha: float, seed: int) -> List[List[int]]:
    if partition == "iid":
        return iid_partition(train, num_clients, seed)
    if partition == "dirichlet":
        return dirichlet_partition(train, num_clients, dirichlet_alpha, seed)
    raise ValueError(f"Unknown partition '{partition}' (expected 'iid' or 'dirichlet')")


def make_client_subsets(train: Dataset, client_indices: List[List[int]]) -> List[Subset]:
    """Return one Subset per client."""
    return [Subset(train, idx) for idx in client_indices]


def make_test_loader(test: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(test, batch_size=batch_size, shuffle=False)
