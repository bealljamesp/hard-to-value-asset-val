# hard-to-value-asset-val/src/val/models/pricing.py
"""Vectorized Discounted Cash Flow (DCF) pricing kernel for illiquid asset Monte Carlo simulations."""

import numpy as np
import numpy.typing as npt


class DCFPricingKernel:
    """Vectorized Discounted Cash Flow pricing kernel for evaluating illiquid asset paths
    using continuous or discrete discount factors.
    """

    def __init__(self, discount_factors: npt.NDArray[np.float64]) -> None:
        # Ensure contiguous memory layout for high-performance matrix operations
        self.discount_factors = np.ascontiguousarray(discount_factors, dtype=np.float64)

    def price_paths(
        self, cash_flows: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the Present Value (PV) for each Monte Carlo path simultaneously.

        Args:
            cash_flows: NDArray of shape (num_paths, num_steps + 1)

        Returns:
            present_values: NDArray of shape (num_paths,) representing path-wise NPVs.
        """
        # Forced vectorization: dot product via native matrix operator (@) across rows
        # cash_flows shape: (P, T), discount_factors shape: (T,) -> broadcast multiply and sum along axis 1
        discounted_cash_flows = cash_flows * self.discount_factors[None, :]

        # O(1) memory footprint reduction via axis summation
        present_values = np.sum(discounted_cash_flows, axis=1)
        return present_values

    def compute_risk_metrics(
        self, present_values: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """Compute core portfolio valuation risk metrics (Expected Value, Value-at-Risk, Expected Shortfall)."""
        mean_val = float(np.mean(present_values))
        std_val = float(np.std(present_values))

        # 95% and 99% Value-at-Risk (parametric / historical hybrid on simulation paths)
        var_95 = float(np.percentile(present_values, 5.0))
        var_99 = float(np.percentile(present_values, 1.0))

        # Expected Shortfall (Conditional VaR at 95%)
        es_95 = (
            float(np.mean(present_values[present_values <= var_95]))
            if np.any(present_values <= var_95)
            else var_95
        )

        return {
            "mean_npv": mean_val,
            "std_npv": std_val,
            "var_95": var_95,
            "var_99": var_99,
            "es_95": es_95,
        }
