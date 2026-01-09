from __future__ import annotations

"""Network topology construction utilities for the social graph."""

import networkx as nx
from typing import Literal, Optional
import random


Topology = Literal["erdos_renyi", "small_world", "scale_free", "community"]


def build_network(
    n: int,
    topology: Topology,
    seed: int = 42,
    p: float = 0.05,         # Edge probability / rewiring probability.
    k: int = 4,              # Number of neighbours in the small-world graph.
    m: int = 2,              # Number of new edges per node in the scale-free graph.
    communities: int = 4,
) -> nx.Graph:
    """Construct a NetworkX graph with a specified topology.

    The resulting graph represents the social network of investors and
    is used to model information diffusion and social influence. Each
    edge is initialised with a trust weight that can subsequently evolve
    in response to agent performance.
    """
    rng = random.Random(seed)

    if topology == "erdos_renyi":
        G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    elif topology == "small_world":
        # Watts–Strogatz small-world graph.
        G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    elif topology == "scale_free":
        # Barabási–Albert preferential-attachment graph.
        G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    elif topology == "community":
        # Simple stochastic block model with denser within-community connectivity.
        sizes = [n // communities] * communities
        sizes[0] += n - sum(sizes)
        # Probability matrix with higher intra-community than inter-community density.
        pin, pout = min(0.35, 1.0), max(0.02, p)
        probs = [[pin if i == j else pout for j in range(communities)] for i in range(communities)]
        G = nx.stochastic_block_model(sizes, probs, seed=seed)
    else:
        raise ValueError(f"Unknown topology={topology}")

    # Ensure minimum connectivity by attaching isolated nodes to random neighbours.
    isolates = list(nx.isolates(G))
    if isolates:
        nodes = list(G.nodes())
        for u in isolates:
            v = rng.choice([x for x in nodes if x != u])
            G.add_edge(u, v)

    # Initialise influence/trust weights on all edges.
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(0.2, 1.0)   # Trust/influence weight.
    return G
