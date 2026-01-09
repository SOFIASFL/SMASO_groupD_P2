from __future__ import annotations

"""Investor agent implementing a full agentic decision loop.

The ``InvestorAgent`` represents a heterogeneous market participant with
an explicit portfolio, risk preferences, and sensitivity to both local
social influence and global analyst recommendations. Its decision
process follows the agentic sequence Observe–Recall–Plan–Act–Reflect–
Update, with a bounded episodic memory enabling adaptive behaviour
based on past performance and network feedback.
"""

from typing import Optional

from ..core.types import Observation, Plan, Action, Outcome, ActionType
from .base import BaseAgent


class InvestorAgent(BaseAgent):
    """Heterogeneous investor with portfolio, memory, and social influence.

    Investor agents observe market prices and returns, weighted
    neighbour actions from the social network, and the latest analyst
    recommendation. They then recall a short summary of recent
    decisions, plan an action influenced by momentum, social pressure,
    and risk aversion, execute trades during the settlement stage, and
    finally reflect on realised profit and loss to update their memory.
    """

    def __init__(
        self,
        unique_id: int,
        model,
        profile: str,
        risk_aversion: float,
        init_cash: float = 10_000.0,
        memory_capacity: int = 50,
        **kwargs
    ) -> None:
        """Initialise a new investor agent.

        Parameters
        ----------
        unique_id:
            Unique identifier used by Mesa and the social network.
        model:
            Reference to the containing Mesa model.
        profile:
            High-level label describing the investor type (for example,
            risk-averse, moderate, speculative).
        risk_aversion:
            Scalar in ``[0, 1]`` controlling how strongly the agent
            scales down aggressive positions.
        init_cash:
            Initial cash endowment used to finance trades.
        memory_capacity:
            Maximum number of decision episodes stored in memory.
        """
        super().__init__(unique_id, model, memory_capacity)

        self._analyst_plan = None

        self.profile = profile
        self.risk_aversion = float(risk_aversion)

        # Portfolio state: cash and risky-asset holdings.
        self.cash = float(init_cash)
        self.shares = 0.0

        # Bookkeeping variables for the agentic loop across scheduler stages.
        self._pending_action: Optional[Action] = None
        self._last_obs: Optional[Observation] = None
        self._last_plan: Optional[Plan] = None
        self._outcome: Optional[Outcome] = None

        # Last mark-to-market wealth used to compute incremental PnL.
        self._last_wealth = self._mark_to_market()

    # -------------------------
    # Helpers
    # -------------------------
    def _mark_to_market(self) -> float:
        """Compute current wealth by marking the portfolio to market."""
        price = self.model.market.price
        return self.cash + self.shares * price

    # -------------------------
    # Stage 1: decide
    # -------------------------
    def decide(self) -> None:
        """Execute the Observe–Recall–Plan–Act phases of the agentic loop."""
        obs = self.observe()
        recalled = self.recall(obs)
        plan = self.plan(obs, recalled)
        action = self.act(plan)

        # Cache the episode for the subsequent reflection stage.
        self._last_obs = obs
        self._last_plan = plan
        self._pending_action = action

        # Publish the discrete action for social diffusion on the network.
        self.model.last_actions[self.unique_id] = action.action

    def observe(self) -> Observation:
        """Construct an observation combining market, network, and analyst signals."""
        env = self.model.market
        neighbor_signals = self.model.get_neighbor_signals(self.unique_id)
        
        analyst_signal = self.model.get_latest_analyst_signal()  # text
        self._analyst_plan = getattr(self.model, "analyst_recommendation", None)  # structured Plan
        
        return Observation(
            t=self.model.schedule.time,
            price=env.price,
            last_return=env.last_return,
            neighbor_signals=neighbor_signals,
            analyst_signal=analyst_signal,
        )

    def plan(self, obs: Observation, recalled: str) -> Plan:
        """Map observations and memory into a discrete trading plan."""
        
        # 1. Momentum Component
        if obs.last_return > 0:
            momentum = 1.0
        elif obs.last_return < 0:
            momentum = -1.0
        else:
            momentum = 0.0

        # 2. Social Component
        buy_p = obs.neighbor_signals.get(ActionType.BUY, 0.0)
        sell_p = obs.neighbor_signals.get(ActionType.SELL, 0.0)
        social = buy_p - sell_p

        # 3. Analyst Component
        analyst_val = 0.0
        signal_action = None
        if self._analyst_plan is not None:
            signal_action = self._analyst_plan.intended_action

            
            if signal_action == ActionType.BUY:
                analyst_val = 1.0
            elif signal_action == ActionType.SELL:
                analyst_val = -1.0

        # Combined score
        score = 0.3 * momentum + 0.2 * social + 0.5 * analyst_val

        # Risk control: more risk-averse agents downscale the score.
        score *= (1.0 - self.risk_aversion)

        # Map the continuous score to a discrete trading intention.
        if score > 0.15:
            intended = ActionType.BUY
        elif score < -0.15:
            intended = ActionType.SELL
        else:
            intended = ActionType.HOLD

        confidence = min(1.0, max(0.05, abs(score)))

        rationale = (
            f"momentum={momentum:.2f}, social={social:.2f}, analyst={analyst_val:.2f}, "
            f"risk_aversion={self.risk_aversion:.2f}, score={score:.2f}\n"
            f"recent_memory:\n{recalled}"
        )

        return Plan(
            intended_action=intended,
            confidence=confidence,
            rationale=rationale,
            meta={
                "score": score,
                "momentum": momentum,
                "social": social,
                "analyst_val": analyst_val,
                "buy_pressure": buy_p,
                "sell_pressure": sell_p,
            },
        )

    def act(self, plan: Plan) -> Action:
        """Translate the plan into a position size while enforcing risk limits."""
        # Base size scales with confidence, subject to hard bounds.
        size = min(0.5, 0.1 + 0.4 * plan.confidence)  # 0.1..0.5

        # Risk aversion further attenuates aggressiveness.
        size *= (1.0 - 0.5 * self.risk_aversion)
        size = max(0.0, min(0.5, size))

        return Action(
            action=plan.intended_action,
            size=size,
            rationale=plan.rationale,
        )

    # -------------------------
    # Stage 2: market
    # -------------------------
    def market(self) -> None:
        """Market-stage hook for the investor."""
        pass

    # -------------------------
    # Stage 3: settle (execute trade at the post-update price)
    # -------------------------
    def settle(self) -> None:
        """Execute the pending trade and compute realised profit and loss."""
        if self._pending_action is None:
            return

        price = self.model.market.price
        a = self._pending_action

        # Execute trade according to the discrete action and position size.
        if a.action == ActionType.BUY and self.cash > 0 and a.size > 0:
            spend = self.cash * a.size
            qty = spend / price
            self.cash -= spend
            self.shares += qty

        elif a.action == ActionType.SELL and self.shares > 0 and a.size > 0:
            qty = self.shares * a.size
            self.shares -= qty
            self.cash += qty * price

        # Compute incremental PnL and record the outcome for reflection.
        wealth = self._mark_to_market()
        pnl = wealth - self._last_wealth
        self._outcome = Outcome(
            t=self.model.schedule.time,
            pnl=pnl,
            new_wealth=wealth,
            price=price,
        )
        self._last_wealth = wealth

    # -------------------------
    # Stage 4: reflect (Reflect and update memory)
    # -------------------------
    def reflect(self) -> None:
        """Complete the agentic loop by reflecting and updating memory."""
        obs = self._last_obs
        plan = self._last_plan
        action = self._pending_action
        outcome = self._outcome

        if obs is None or plan is None or action is None or outcome is None:
            return

        reflection = self.reflect_text(obs, plan, action, outcome)
        self.update(obs, plan, action, outcome, reflection)

        # Clear per-step buffers before the next decision cycle.
        self._pending_action = None
        self._last_obs = None
        self._last_plan = None
        # NOTE: keep _outcome for one global step so MarketModel can update trust weights
        # self._outcome = None

    def reflect_text(self, obs: Observation, plan: Plan, action: Action, outcome: Outcome) -> str:
        """Generate a concise textual reflection on the realised outcome."""
        good = outcome.pnl >= 0
        return (
            f"{'GOOD' if good else 'BAD'} decision. pnl={outcome.pnl:.2f}. "
            f"Action={action.action}, size={action.size:.2f}. "
            f"Consider adjusting thresholds or position sizing under high volatility."
        )