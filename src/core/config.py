"""Configuration structures for simulation experiments."""

from dataclasses import dataclass
from typing import Literal

Topology = Literal["erdos_renyi", "small_world", "scale_free", "community"]


@dataclass
class SimConfig:
    """High-level configuration for a single simulation run.

    The configuration captures both network-generation parameters and
    core market-process parameters. It is designed to be serialisable
    and easy to extend for experimental studies.
    """

    seed: int = 42
    n_investors: int = 50

    topology: Topology = "small_world"
    p: float = 0.05      # Erdős–Rényi edge probability / rewiring probability.
    k: int = 4           # Number of neighbours in the small-world graph.
    m: int = 2           # Number of new edges per node in the scale-free graph.
    communities: int = 4

    # Market parameters for the GBM-style price process.
    init_price: float = 100.0
    mu: float = 0.0002
    sigma: float = 0.01
