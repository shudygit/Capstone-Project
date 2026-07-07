"""fedblock (baseline module): Federated Averaging (FedAvg) on MNIST.

This is the first module of the capstone project "Hybrid Blockchain-Assisted
Poisoning Detection in Federated Learning". It implements the clean federated
learning baseline - simulated clients training a small CNN on MNIST under both
IID and non-IID (Dirichlet) data partitioning, aggregated with FedAvg - against
which all later modules (poisoning attacks, blockchain ledger, z-score defence)
will be evaluated.

Author: Shudhatm Jain (25253301), University of Galway.
"""

__version__ = "0.1.0"
