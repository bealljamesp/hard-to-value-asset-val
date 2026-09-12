# hard-to-value-asset-val/src/val/models/stress.py
"""Vectorized stress-testing module for illiquid asset portfolios."""

import numpy as np
import numpy.typing as npt

from val.models.cashflows import IlliquidCashFlowSimulator
from val.models.pricing import DCFPricingKernel


class PortfolioStressTester:
    """Vectorized stress-testing harness for evaluating illiquid asset portfolios under extreme market shocks."""

    def __init__(
        self,
        initial_balance: float,
        coupon_rate: float,
        time_horizon: float,
        num_steps: int,
        num_paths: int,
    ) -> None:
        self.initial_balance = float(initial_balance)
        self.coupon_rate = float(coupon_rate)
        self.time_horizon = float(time_horizon)
        self.num_steps = int(num_steps)
        self.num_paths = int(num_paths)

    def run_stress_grid(
        self,
        default_intensity_shocks: npt.NDArray[np.float64],
        volatility_shocks: npt.NDArray[np.float64],
        pricing_kernel: DCFPricingKernel,
        time_grid: npt.NDArray[np.float64],
        seed: int = 42,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Simulate portfolio NPV risk metrics across a multi-dimensional stress grid
        of default intensity and volatility shocks.
        """
        num_di = default_intensity_shocks.size
        num_vol = volatility_shocks.size

        mean_npvs = np.zeros((num_di, num_vol), dtype=np.float64)
        var_95_grid = np.zeros((num_di, num_vol), dtype=np.float64)

        # Vectorized grid traversal via broadcasted evaluations
        for i, di in enumerate(default_intensity_shocks):
            for j, vol in enumerate(volatility_shocks):
                simulator = IlliquidCashFlowSimulator(
                    initial_balance=self.initial_balance,
                    coupon_rate=self.coupon_rate,
                    default_intensity=di,
                    volatility=vol,
                    time_horizon=self.time_horizon,
                    num_steps=self.num_steps,
                    num_paths=self.num_paths,
                    seed=seed + i * 100 + j,
                )
                _, cash_flows = simulator.simulate_paths()
                npvs = pricing_kernel.price_paths(cash_flows, time_grid)
                metrics = pricing_kernel.compute_risk_metrics(npvs)

                mean_npvs[i, j] = metrics["mean_npv"]
                var_95_grid[i, j] = metrics["var_95"]

        return {
            "mean_npvs": mean_npvs,
            "var_95_grid": var_95_grid,
        }
