from __future__ import annotations

"""Lightweight episodic memory for agentic behaviour."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional
from .types import MemoryItem


@dataclass
class AgentMemory:
    """Bounded queue of decision episodes for an individual agent.

    ``AgentMemory`` stores recent decision episodes as ``MemoryItem``
    instances in a fixed-capacity deque. It supports retrieval of the
    latest episode, enumeration of all stored items, and generation of
    concise textual summaries for use by heuristic planners or LLMs.
    """

    capacity: int = 50

    def __post_init__(self) -> None:
        """Initialise the underlying bounded deque."""
        self._items: Deque[MemoryItem] = deque(maxlen=self.capacity)

    def add(self, item: MemoryItem) -> None:
        """Append a new decision episode to memory."""
        self._items.append(item)

    def last(self) -> Optional[MemoryItem]:
        """Return the most recently stored episode, if any."""
        return self._items[-1] if self._items else None

    def items(self) -> List[MemoryItem]:
        """Return a list view over all stored episodes."""
        return list(self._items)

    def summarize(self, k: int = 5) -> str:
        """Summarise the last ``k`` episodes as a compact text string.

        The summary is intentionally terse and structured so that it can
        be used as context for an LLM or other downstream components
        without exposing the full memory contents.
        """
        tail = list(self._items)[-k:]
        if not tail:
            return "No prior decisions."
        lines = []
        for it in tail:
            lines.append(
                f"t={it.t} act={it.action.action} size={it.action.size:.2f} pnl={it.outcome.pnl:.2f} "
                f"wealth={it.outcome.new_wealth:.2f} conf={it.plan.confidence:.2f}"
            )
        return "\n".join(lines)
