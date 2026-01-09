"""Custom Mesa scheduler enforcing a staged causal order.

The ``StagedScheduler`` decomposes each global simulation step into
four ordered phases: ``decide``, ``market``, ``settle``, and
``reflect``. This separation makes the causal structure of the model
transparent and ensures that agents first commit to actions, then
experience market updates, then settle portfolios, and only afterwards
engage in reflective learning.
"""

from mesa.time import BaseScheduler


class StagedScheduler(BaseScheduler):
    """Scheduler that iterates agents through named behavioural stages."""

    stages = ("decide", "market", "settle", "reflect")

    def step(self) -> None:
        """Advance the model by one step across all defined stages.

        For each stage, the scheduler first executes any global model
        hook (currently, a market-wide price update in the ``market``
        stage), and then iterates over all agents, invoking the method
        whose name matches the current stage if it is defined. This
        pattern supports agentic loops where decision, execution, and
        learning are cleanly separated in time.
        """
        for stage in self.stages:
            # Invoke a single global market hook once per step.
            if stage == "market":
                if hasattr(self.model, "market_global_update"):
                    self.model.market_global_update()

            # Invoke the stage-specific method on each agent, if present.
            for agent in list(self.agents):
                method = getattr(agent, stage, None)
                if callable(method):
                    method()

        self.steps += 1
        self.time += 1
