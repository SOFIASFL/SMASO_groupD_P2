from __future__ import annotations

"""Base Mesa agent integrating the generic agentic loop.

This module defines ``BaseAgent``, a small adapter that combines Mesa's
``Agent`` class with the abstract ``AgenticLoop`` interface and the
concrete ``AgentMemory`` implementation. Concrete agents such as
``InvestorAgent`` and ``AnalystLLMAgent`` inherit from this class to
gain access to the full Observe–Recall–Plan–Act–Reflect–Update
behavioural cycle and an episode-based memory.
"""

from mesa import Agent
from ..core.agentic import AgenticLoop
from ..core.memory import AgentMemory


class BaseAgent(Agent, AgenticLoop):
    """Base class for all agents in the model.

    ``BaseAgent`` ensures that every agent participating in the market
    is simultaneously a Mesa ``Agent`` (for scheduling and interaction)
    and an ``AgenticLoop`` entity (for cognitive structure and memory).
    The constructor wires a bounded ``AgentMemory`` into the agentic
    loop, enabling systematic storage and recall of decision episodes.
    """

    def __init__(self, unique_id: int, model, memory_capacity: int = 50) -> None:
        """Initialise the agent with a bounded memory."""
        Agent.__init__(self, unique_id, model)
        AgenticLoop.__init__(self, memory=AgentMemory(capacity=memory_capacity))
