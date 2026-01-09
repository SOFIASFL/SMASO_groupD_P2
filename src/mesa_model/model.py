from __future__ import annotations

"""Mesa model for a stylised, networked financial market.

The ``MarketModel`` couples heterogeneous ``InvestorAgent`` and
``AnalystLLMAgent`` instances with a stochastic market environment and a
NetworkX social network. A staged scheduler enforces a causal ordering
between agent decisions, market price updates, portfolio settlement, and
reflective learning, enabling the study of emergent dynamics driven by
agentic behaviour, memory, and network effects.
"""

from mesa import Model
import networkx as nx
from typing import Dict, Optional, Any, List

from .schedule import StagedScheduler
from ..market.environment import MarketEnvironment
from ..network.influence import neighbor_action_distribution
from ..network.metrics import compute_network_metrics
from ..core.types import ActionType
from ..agents.investor import InvestorAgent
from ..agents.analyst import AnalystLLMAgent

from ..network.evolution import update_trust_weights


class MarketModel(Model):
    """Agent-based financial market model with social structure.

    Parameters
    ----------
    G:
        NetworkX graph representing the investor social network. Nodes
        are aligned with investor identifiers and edges encode channels
        of social influence (with optional trust weights).
    n_investors:
        Number of investor agents to instantiate. Investors are mapped
        to the first ``n_investors`` nodes of ``G``.
    seed:
        Random seed used to initialise the market environment and any
        other stochastic components.
    """

    def __init__(self, G: nx.Graph, n_investors: int, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed
        self.network = G
        self.schedule = StagedScheduler(self)

        # Market environment for the traded asset.
        self.market = MarketEnvironment(seed=seed)

        # Shared state used by agents to coordinate decisions.
        self.last_actions: Dict[int, ActionType] = {}
        self.latest_analyst_signal: Optional[str] = None

        # Time series used for ex post analysis of prices and network state.
        self.price_history: List[float] = []
        self.net_metrics_history: List[Dict[str, Any]] = []

        # Single global analyst agent that is not embedded in the social graph.
        analyst = AnalystLLMAgent(unique_id=10_000, model=self)
        self.schedule.add(analyst)

        # Create investors aligned with graph nodes 0..n-1.
        for i in range(n_investors):
            profile, ra = self._assign_profile(i)
            inv = InvestorAgent(unique_id=i, model=self, profile=profile, risk_aversion=ra)
            self.schedule.add(inv)
            self.last_actions[i] = ActionType.HOLD

        # Validate that investor identifiers are a subset of the network nodes.
        assert set(range(n_investors)).issubset(self.network.nodes()), (
            "Investor IDs (0..n-1) do not match the graph nodes."
        )

        # Initial network metrics prior to any agent decisions.
        self.net_metrics = compute_network_metrics(self.network)

        # Record the initial state for downstream analysis.
        self.price_history.append(self.market.price)
        self.net_metrics_history.append({
            "t": self.schedule.time,
            "avg_degree": self.net_metrics["avg_degree"],
            "avg_clustering": self.net_metrics["avg_clustering"],
            "modularity": self.net_metrics["modularity"],
        })

    def _assign_profile(self, i: int):
        """Assign a simple behavioural profile and risk aversion parameter.

        The current implementation creates three stylised investor types
        (risk-averse, moderate, speculative) based on the agent index.
        This is intentionally lightweight and can be replaced by richer
        profiling schemes without affecting the surrounding architecture.
        """
        if i % 3 == 0:
            return "risk_averse", 0.8
        if i % 3 == 1:
            return "moderate", 0.5
        return "speculative", 0.2

    def get_neighbor_signals(self, node_id: int):
        """Return the distribution of neighbour actions for a given investor.

        The distribution is computed as a weighted histogram over the
        last actions of the investor's network neighbours, where edge
        weights in the NetworkX graph encode influence or trust.
        """
        return neighbor_action_distribution(self.network, node_id, self.last_actions)

    def get_latest_analyst_signal(self) -> Optional[str]:
        """Return the most recent textual recommendation from the analyst."""
        return self.latest_analyst_signal

    def step(self) -> None:
        """Advance the simulation by one full scheduler step.

        The staged scheduler drives agent-level decision making and
        learning, after which this method updates global network metrics,
        records time series for analysis, and evolves trust weights based
        on realised investor performance.
        """
        self.schedule.step()

        # (A) Update network-level metrics derived from the current graph.
        self.net_metrics = compute_network_metrics(self.network)

        # (B) Append price and network metrics to the simulation history.
        self.price_history.append(self.market.price)
        self.net_metrics_history.append({
            "t": self.schedule.time,
            "avg_degree": self.net_metrics["avg_degree"],
            "avg_clustering": self.net_metrics["avg_clustering"],
            "modularity": self.net_metrics["modularity"],
        })

        # (C) Update trust weights based on investors' step-level PnL
        #     to obtain an evolving influence network.
        pnl_map: Dict[int, float] = {}
        for a in self.schedule.agents:
            # Only investor agents have a non-trivial Outcome with PnL;
            # the analyst contributes a zero or missing PnL.
            if hasattr(a, "_outcome") and getattr(a, "_outcome") is not None:
                pnl_map[a.unique_id] = a._outcome.pnl

        update_trust_weights(self.network, pnl_map)

    def market_global_update(self) -> None:
        """Hook invoked by the scheduler during the 'market' stage.

        Aggregates investor order flow into a single net flow and
        advances the market environment, thereby closing the loop between
        decentralised trading decisions and price dynamics.
        """
        net_flow = self._compute_net_order_flow()
        self.market.step(net_flow)

    def _compute_net_order_flow(self) -> float:
        """Compute net order flow, weighted by network centrality.

        The current mechanism interprets investor actions as unit buy or
        sell impulses and scales them by degree centrality plus a small
        baseline. This makes highly connected agents exert stronger
        price impact, reflecting their central position in the social
        network.
        """
        deg_cent = self.net_metrics["degree_centrality"]
        flow = 0.0
        for node, a in self.last_actions.items():
            w = float(deg_cent.get(node, 0.0)) + 0.1
            if a == ActionType.BUY:
                flow += w
            elif a == ActionType.SELL:
                flow -= w
        return flow
