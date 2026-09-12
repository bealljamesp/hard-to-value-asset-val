# hard-to-value-asset-val/src/val/models/cashflows.py
"""Vectorized Monte Carlo simulator for illiquid asset cash flows."""

import numpy as np
import numpy.typing as npt


class IlliquidCashFlowSimulator:
    """Vectorized Monte Carlo simulator for stochastic cash flows of hard-to-value assets.
    Executes entirely within a deterministic NumPy space using C-contiguous arrays.
    """

    def __init__(
        self,
        initial_balance: float,
        coupon_rate: float,
        default_intensity: float,
        volatility: float,
        time_horizon: float,
        num_steps: int,
        num_paths: int,
        seed: int | None = None,
    ) -> None:
        self.initial_balance = float(initial_balance)
        self.coupon_rate = float(coupon_rate)
        self.default_intensity = float(default_intensity)
        self.volatility = float(volatility)
        self.time_horizon = float(time_horizon)
        self.num_steps = int(num_steps)
        self.num_paths = int(num_paths)

        self.dt = self.time_horizon / self.num_steps
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Simulate asset balance and cash-flow paths simultaneously using vectorized Geometric Brownian Motion
        with jump-to-default intensity.

        Returns:
            balances: NDArray of shape (num_paths, num_steps + 1)
            cash_flows: NDArray of shape (num_paths, num_steps + 1)
        """
        # Generate standard normal shocks for all paths and steps simultaneously: O(N * M) matrix operations
        z = self.rng.standard_normal((self.num_paths, self.num_steps))

        # Time grid configuration
        t_grid = np.linspace(0.0, self.time_horizon, self.num_steps + 1)

        # Initialize balance matrix (C-contiguous layout)
        balances = np.zeros(
            (self.num_paths, self.num_steps + 1), dtype=np.float64, order="C"
        )
        balances[:, 0] = self.initial_balance

        # Vectorized drift and diffusion factors
        drift = (self.risk_free_rate_adjustment() - 0.5 * self.volatility**2) * self.dt
        diffusion = self.volatility * np.sqrt(self.dt)

        # Vectorized path evolution across time steps without Python loops
        for i in range(self.num_steps):
            # GBM step: B_{t+dt} = B_t * exp(drift + diffusion * Z)
            shocks = drift + diffusion * z[:, i]
            balances[:, i + 1] = balances[:, i] * np.exp(shocks)

        # Simulate default states using exponential distribution for default times
        default_times = self.rng.exponential(
            1.0 / self.default_intensity, size=self.num_paths
        )

        # Zero out balances post-default via broadcast comparison against the time grid
        default_mask = t_grid[None, :] > default_times[:, None]
        balances[default_mask] = 0.0

        # Compute cash flows: coupons on active balances + principal adjustments
        coupon_flows = balances[:, :-1] * self.coupon_rate * self.dt
        principal_deltas = -np.diff(balances, axis=1)
        cash_flows = coupon_flows + np.maximum(principal_deltas, 0.0)

        # Pad cash flows to match time grid shape (num_paths, num_steps + 1)
        initial_zero_cf = np.zeros((self.num_paths, 1), dtype=np.float64)
        cash_flows = np.hstack((initial_zero_cf, cash_flows))

        return balances, cash_flows

    def risk_free_rate_adjustment(self) -> float:
        return 0.035  # Baseline drift component placeholder
