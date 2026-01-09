from __future__ import annotations

"""Social influence utilities operating on the investor network."""

import networkx as nx
from typing import Dict
from ..core.types import ActionType


def neighbor_action_distribution(G: nx.Graph, node: int, last_actions: Dict[int, ActionType]) -> Dict[ActionType, float]:
    """Compute the weighted distribution of neighbour actions for a node.

    The function aggregates the latest discrete actions of a node's
    neighbours, weighting each contribution by the trust weight on the
    corresponding edge. The resulting normalised distribution serves as
    a compact representation of local social pressure on the agent.
    """
    totals = {ActionType.BUY: 0.0, ActionType.SELL: 0.0, ActionType.HOLD: 0.0}
    wsum = 0.0

    for nbr in G.neighbors(node):
        w = float(G[node][nbr].get("weight", 1.0))
        a = last_actions.get(nbr, ActionType.HOLD)
        totals[a] += w
        wsum += w

    if wsum <= 1e-12:
        return {k: 0.0 for k in totals}

    return {k: v / wsum for k, v in totals.items()}
