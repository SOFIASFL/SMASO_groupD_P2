from __future__ import annotations

"""Computation of descriptive metrics for the social network."""

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
from typing import Dict, Any


def compute_network_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute centrality, clustering, and modularity metrics for a graph.

    The returned dictionary includes both aggregate statistics (such as
    average degree and average clustering coefficient) and node-level
    quantities (degree centrality, local clustering, detected
    communities). These metrics are used to analyse how network
    structure relates to herding, contagion, and other emergent market
    phenomena.
    """
    deg_cent = nx.degree_centrality(G)
    clustering = nx.clustering(G, weight="weight")

    # Communities identified via greedy modularity maximisation.
    comms = list(greedy_modularity_communities(G, weight="weight"))
    mod = modularity(G, comms, weight="weight") if comms else 0.0

    return {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / max(1, G.number_of_nodes()),
        "avg_degree_centrality": sum(deg_cent.values()) / max(1, len(deg_cent)),
        "avg_clustering": sum(clustering.values()) / max(1, len(clustering)),
        "modularity": mod,
        "degree_centrality": deg_cent,
        "clustering": clustering,
        "communities": comms,
    }
