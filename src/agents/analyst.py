import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from ..core.types import Observation, Plan, Action, Outcome, ActionType
from .base import BaseAgent


class AnalystLLMAgent(BaseAgent):
    """
    Analyst agent powered by Groq (LLaMA 3) as an OPTIONAL advisory component.

    - When the Groq API / key is available, the agent produces a structured recommendation (BUY/SELL/HOLD)
      with confidence and reasoning in JSON.
    - When unavailable, the agent falls back to a neutral HOLD recommendation.
    - The analyst does NOT trade; it only publishes advice and still completes the agentic loop.
    """

    def __init__(self, unique_id: int, model, memory_capacity: int = 50):
        super().__init__(unique_id, model, memory_capacity)
        load_dotenv()

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.llm_model = "llama-3.3-70b-versatile"

        # Buffers to complete the agentic loop across stages
        self._last_obs: Observation | None = None
        self._last_plan: Plan | None = None
        self._published_action: Action | None = None
        self._outcome: Outcome | None = None

    # -------------------------
    # Stage 1: decide
    # -------------------------
    def decide(self) -> None:
        obs = self.observe()
        recalled = self.recall(obs)
        plan = self.plan(obs, recalled)
        action = self.act(plan)

        # store for reflection stage
        self._published_action = action

    def observe(self) -> Observation:
        """
        The analyst observes ONLY market-level state.
        It does not use social neighbour signals (set to zeros).
        """
        base_obs = self.model.get_observation()

        return Observation(
            t=base_obs.t,
            price=base_obs.price,
            last_return=base_obs.last_return,
            neighbor_signals={
                ActionType.BUY: 0.0,
                ActionType.SELL: 0.0,
                ActionType.HOLD: 0.0,
            },
            analyst_signal=None,
        )

    def plan(self, obs: Observation, recalled: str) -> Plan:
        """
        Call Groq for a structured JSON recommendation when available.
        Fallback to HOLD if API fails or key is missing.
        """
        prompt = (
            "Context: You are a swing trader.\n"
            f"Market Data: Price is {obs.price:.2f}. Last change was {obs.last_return:.4f}.\n"
            f"Recent memory summary:\n{recalled}\n\n"
            "Strategy:\n"
            "1. If Last change was POSITIVE (price went up), consider SELLING (Take Profit).\n"
            "2. If Last change was NEGATIVE (price went down), consider BUYING (Buy Dip).\n"
            "3. If change is tiny, HOLD.\n\n"
            "Respond ONLY JSON: {\"action\": \"BUY\", \"confidence\": 0.9, \"reasoning\": \"reason\"}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)

            plan = Plan(
                intended_action=ActionType(data["action"]),
                confidence=float(data["confidence"]),
                rationale=data["reasoning"],
                meta={"source": "groq"},
            )

        except Exception as e:
            plan = Plan(
                intended_action=ActionType.HOLD,
                confidence=0.5,
                rationale=f"[FALLBACK] HOLD. Reason: {e}",
                meta={"source": "fallback"},
            )

        # store for reflection
        self._last_obs = obs
        self._last_plan = plan
        return plan

    def act(self, plan: Plan) -> Action:
        """
        Publish the analyst recommendation to the model.
        Returns an Action object (size=0.0) to complete the AgenticLoop contract.
        """
        action = Action(action=plan.intended_action, size=0.0, rationale=plan.rationale)

        # Publish BOTH:
        # (A) text signal used by investors via model.get_latest_analyst_signal()
        self.model.latest_analyst_signal = plan.rationale

        # (B) structured plan (optional; useful for debugging/analysis)
        self.model.analyst_recommendation = plan

        return action

    # -------------------------
    # Stage 2: market
    # -------------------------
    def market(self) -> None:
        # Analyst does not intervene in market price formation
        pass

    # -------------------------
    # Stage 3: settle
    # -------------------------
    def settle(self) -> None:
        # Analyst has no portfolio; still produce an outcome to close the loop
        self._outcome = Outcome(
            t=getattr(self.model.schedule, "time", 0),
            pnl=0.0,
            new_wealth=0.0,
            price=self.model.market.price,
        )

    # -------------------------
    # Stage 4: reflect
    # -------------------------
    def reflect(self) -> None:
        if self._last_obs is None or self._last_plan is None or self._published_action is None or self._outcome is None:
            return

        reflection = "Delivered advisory recommendation (Groq if available; fallback otherwise)."
        self.update(self._last_obs, self._last_plan, self._published_action, self._outcome, reflection)

        # clear buffers
        self._last_obs = None
        self._last_plan = None
        self._published_action = None
        self._outcome = None