from __future__ import annotations

"""Typed data structures used throughout the agentic model."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List


class ActionType(str, Enum):
    """Discrete action types available to investor and analyst agents."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Observation:
    """Snapshot of information available to an agent at decision time."""

    t: int
    price: float
    last_return: float
    neighbor_signals: Dict[ActionType, float]
    analyst_signal: Optional[str] = None       # Textual recommendation from the analyst.


@dataclass(frozen=True)
class Plan:
    """Internal representation of an agent's intended action."""

    intended_action: ActionType
    confidence: float
    rationale: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Action:
    """Executable trading or signalling action derived from a plan."""

    action: ActionType
    size: float          # Fraction of cash/position to adjust (0..1).
    rationale: str


@dataclass(frozen=True)
class Outcome:
    """Realised financial outcome of executing an action."""

    t: int
    pnl: float
    new_wealth: float
    price: float


@dataclass(frozen=True)
class MemoryItem:
    """Complete record of a single decision episode for memory storage."""

    t: int
    observation: Observation
    plan: Plan
    action: Action
    outcome: Outcome
    reflection: str
    tags: List[str]
