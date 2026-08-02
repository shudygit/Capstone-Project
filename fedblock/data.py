"""MNIST loading and IID / non-IID (Dirichlet) client partitioning."""
from __future__ import annotations

import gzip
import os
import shutil
from typing import List

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# MNIST channel statistics used by virtually all reference implementations.
_MNIST_MEAN, _MNIST_STD = 0.1307, 0.3081

# Project root = the folder containing the `fedblock` package.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# The four raw IDX files torchvision expects inside <root>/MNIST/raw.
_MNIST_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((_MNIST_MEAN,), (_MNIST_STD,)),
    ])


def resolve_data_root(data_root: str):
    """Make ``data_root`` absolute, resolving relative paths against the project root.

    This means experiments work no matter which directory you launch them from.
    """
    if os.path.isabs(data_root):
        return data_root
    return os.path.normpath(os.path.join(_PROJECT_ROOT, data_root))


def _ensure_extracted(raw_dir: str):
    """Decompress any bundled ``.gz`` files that have no extracted counterpart.

    Purely local: it only ever reads ``.gz`` files already on disk.
    """
    for name in _MNIST_FILES:
        target = os.path.join(raw_dir, name)
        archive = target + ".gz"
        if not os.path.exists(target) and os.path.exists(archive):
            with gzip.open(archive, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def load_mnist(data_root: str):
    """Return the MNIST train and test datasets from local files only.

    Raises a clear error if the bundled dataset is missing, rather than silently
    attempting an online download.
    """
    root = resolve_data_root(data_root)
    raw_dir = os.path.join(root, "MNIST", "raw")
    _ensure_extracted(raw_dir)

    missing = [n for n in _MNIST_FILES if not os.path.exists(os.path.join(raw_dir, n))]
    if missing:
        raise FileNotFoundError(
            f"MNIST data not found in {raw_dir}.\n"
            f"Missing files: {missing}\n"
            "The dataset ships with this repository under data/MNIST/raw. "
            "Restore that folder (or copy it from the repo) and re-run. "
            "No download is attempted by design."
        )

    tfm = _transform()
    # download=False guarantees torchvision never opens a network connection.
    train = datasets.MNIST(root, train=True, download=False, transform=tfm)
    test = datasets.MNIST(root, train=False, download=False, transform=tfm)
    return train, test


def iid_partition(train: Dataset, num_clients: int, seed: int):
    """Shuffle and split indices into ``num_clients`` equal IID shards."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    return [shard.tolist() for shard in np.array_split(idx, num_clients)]


def dirichlet_partition(train: Dataset, num_clients: int, alpha: float,
                        seed: int):
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
                   dirichlet_alpha: float, seed: int):
    if partition == "iid":
        return iid_partition(train, num_clients, seed)
    if partition == "dirichlet":
        return dirichlet_partition(train, num_clients, dirichlet_alpha, seed)
    raise ValueError(f"Unknown partition '{partition}' (expected 'iid' or 'dirichlet')")


def make_client_subsets(train: Dataset, client_indices: List[List[int]]):
    return [Subset(train, idx) for idx in client_indices]


def make_test_loader(test: Dataset, batch_size: int):
    return DataLoader(test, batch_size=batch_size, shuffle=False)
