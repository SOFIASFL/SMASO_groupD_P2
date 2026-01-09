import random

"""Evolutionary mechanisms for trust weights and network structure."""

import networkx as nx


def update_trust_weights(
    G: nx.Graph,
    agent_pnl: dict[int, float],
    lr: float = 0.05,
    w_min: float = 0.05,
    w_max: float = 2.0,
):
    """Adapt edge weights based on recent agent performance.

    The update rule increases the trust weight on edges adjacent to
    agents with positive profit-and-loss and decreases it for agents
    with negative performance, while enforcing lower and upper bounds.
    This yields an endogenous, performance-driven evolution of social
    influence in the network.
    """
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        # Symmetric performance signal from the two incident agents.
        signal = 0.5 * (agent_pnl.get(u, 0.0) + agent_pnl.get(v, 0.0))
        direction = 1.0 if signal > 0 else -1.0 if signal < 0 else 0.0
        w = w * (1.0 + lr * direction)
        data["weight"] = max(w_min, min(w_max, w))


def rewire_by_performance(
    G: nx.Graph,
    rng: random.Random,
    agent_score: dict[int, float],
    prob: float = 0.01,
):
    """Optionally rewire edges towards high-performing agents.

    With a small probability per node, the algorithm removes the weakest
    existing connection and creates a new edge to a high-scoring agent
    to capture performance-driven reallocation of attention or trust.
    """
    nodes = list(G.nodes())
    for u in nodes:
        if rng.random() > prob:
            continue
        nbrs = list(G.neighbors(u))
        if not nbrs:
            continue
        # Remove the weakest existing edge for node u.
        v_remove = min(nbrs, key=lambda v: float(G[u][v].get("weight", 1.0)))
        G.remove_edge(u, v_remove)

        # Connect to a high-scoring candidate (not self, not already connected).
        candidates = [x for x in nodes if x != u and not G.has_edge(u, x)]
        if not candidates:
            continue
        v_add = max(candidates, key=lambda x: agent_score.get(x, 0.0))
        G.add_edge(u, v_add, weight=rng.uniform(0.2, 1.0))
