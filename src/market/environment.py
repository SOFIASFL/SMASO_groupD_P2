from __future__ import annotations

"""Stochastic market environment with GBM-style price dynamics."""

from dataclasses import dataclass
import math
import random

@dataclass
class MarketEnvironment:
    """Single-asset market following a Geometric Brownian Motion (GBM) process.
    
    The environment calculates price updates based on statistical drift, 
    volatility, and investor demand (order flow).
    """

    price: float = 100.0
    mu: float = 0.0002    # Drift from configuration
    sigma: float = 0.01   # Volatility from configuration
    dt: float = 1.0       
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialize the random number generator and return state."""
        self.rng = random.Random(self.seed)
        self.last_return = 0.0

    def step(self, net_order_flow: float) -> None:
        """Advance the market price by one time step.
        
        Combines a stochastic GBM term with a linear market impact term 
        derived from net investor activity.
        """
        # Random shock from a standard normal distribution
        z = self.rng.gauss(0.0, 1.0)
        
        # Calculate the GBM component of the return
        gbm_ret = (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * math.sqrt(self.dt) * z
        
        # Calculate market impact from order flow
        impact = 0.001 * net_order_flow 
        
        # Update price and record the realized return
        ret = gbm_ret + impact
        self.last_return = ret
        self.price *= math.exp(ret)