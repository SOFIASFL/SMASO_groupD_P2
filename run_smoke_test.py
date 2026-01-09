"""Smoke test for the agent-based market model with optional LLM verification.

This module instantiates the market model, optionally checks the AnalystLLMAgent
(LLM) availability, and runs a short simulation loop.
"""

from src.core.config import SimConfig
from src.network.topology import build_network
from src.mesa_model.model import MarketModel
from src.agents.analyst import AnalystLLMAgent
from src.core.types import Observation, ActionType


def main():
    """Run a short simulation to validate model wiring."""
    cfg = SimConfig(seed=42, n_investors=30, topology="small_world", p=0.1, k=4)

    G = build_network(
        n=cfg.n_investors,
        topology=cfg.topology,
        seed=cfg.seed,
        p=cfg.p,
        k=cfg.k,
        m=cfg.m,
        communities=cfg.communities,
    )

    model = MarketModel(G=G, n_investors=cfg.n_investors, seed=cfg.seed)

    # -------------------------------------------------------------------------
    # SYSTEM CHECK: validate the existing AnalystLLMAgent (optional)
    # -------------------------------------------------------------------------
    print("\n[SYSTEM] Checking optional LLM analyst...")

    analyst = next((a for a in model.schedule.agents if isinstance(a, AnalystLLMAgent)), None)

    if analyst is None:
        print("[WARNING] No AnalystLLMAgent found in the model scheduler. Skipping LLM check.")
    else:
        try:
            test_obs = Observation(
                t=0,
                price=model.market.price,
                last_return=0.0,
                neighbor_signals={
                    ActionType.BUY: 0.0,
                    ActionType.SELL: 0.0,
                    ActionType.HOLD: 0.0,
                },
                analyst_signal=None,
            )

            print(f"[SYSTEM] Verifying LLM connection (Price: {test_obs.price:.2f})...")
            plan = analyst.plan(test_obs, "System Check")
            print(f"[SUCCESS] Analyst responded: {plan.intended_action} (conf={plan.confidence:.2f})")

        except Exception as e:
            print(f"[WARNING] LLM check failed: {e}")
            print("[INFO] Continuing without live LLM (fallback mode).")

    print("-" * 60)
    # -------------------------------------------------------------------------

    # Run simulation
    for _ in range(10):
        model.step()

    print("\n[OK] Smoke test ok")
    print("Final price:", model.market.price)
    print(
        "Last metrics:",
        {k: model.net_metrics[k] for k in ["avg_degree", "avg_clustering", "modularity"]},
    )


if __name__ == "__main__":
    main()