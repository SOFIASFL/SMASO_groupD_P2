from __future__ import annotations

"""Abstract agentic loop defining the cognitive structure of agents.

The ``AgenticLoop`` mixin specifies the canonical sequence of operations
for all agents in the model: Observe, Recall, Plan, Act, Reflect, and
Update. Concrete agents implement the abstract steps while delegating
episode storage and summarisation to an ``AgentMemory`` instance.
"""

from abc import ABC, abstractmethod
from .types import Observation, Plan, Action, Outcome, MemoryItem
from .memory import AgentMemory


class AgenticLoop(ABC):
    """Abstract base class for agentic behaviour.

    Subclasses define how observations are formed, plans are generated,
    actions are selected, and reflections are produced. The provided
    ``update`` method appends complete decision episodes to memory,
    enabling subsequent recall and adaptive behaviour.
    """

    def __init__(self, memory: AgentMemory) -> None:
        """Initialise the agentic loop with a concrete memory backend."""
        self.memory = memory

    @abstractmethod
    def observe(self) -> Observation: ...

    def recall(self, obs: Observation) -> str:
        """Return a short textual summary of recent memory.

        The default implementation queries the underlying ``AgentMemory``
        for the last ``k`` episodes and is intended to provide a compact
        context that can be fed into an LLM or heuristic planner.
        """
        return self.memory.summarize(k=5)

    @abstractmethod
    def plan(self, obs: Observation, recalled: str) -> Plan: ...

    @abstractmethod
    def act(self, plan: Plan) -> Action: ...

    @abstractmethod
    def reflect(self, obs: Observation, plan: Plan, action: Action, outcome: Outcome) -> str: ...

    def update(self, obs: Observation, plan: Plan, action: Action, outcome: Outcome, reflection: str) -> None:
        """Store a completed decision episode in memory."""
        item = MemoryItem(
            t=obs.t,
            observation=obs,
            plan=plan,
            action=action,
            outcome=outcome,
            reflection=reflection,
            tags=[],
        )
        self.memory.add(item)

    def step_agentic(self) -> Action:
        """Run a full agentic cycle up to the action stage.

        This helper is useful in settings where outcome realisation and
        reflection are handled externally (for example, by the model
        after a market update), while the agent itself only performs the
        forward-looking stages.
        """
        obs = self.observe()
        recalled = self.recall(obs)
        plan = self.plan(obs, recalled)
        action = self.act(plan)
        # Outcome computation and reflection/update are performed by the
        # model after the market environment has been updated.
        return action
