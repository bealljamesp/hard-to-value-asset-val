# hard-to-value-asset-val/src/val/models/pricing.py
"""Vectorized Discounted Cash Flow (DCF) pricing kernel for illiquid asset Monte Carlo simulations."""

import numpy as np
import numpy.typing as npt
import QuantLib as ql


class DCFPricingKernel:
    """Vectorized Discounted Cash Flow pricing kernel integrated with QuantLib term structures
    and illiquidity adjustment mechanics.
    """

    def __init__(
        self, term_structure_handle: ql.YieldTermStructureHandle, evaluation_date: str
    ) -> None:
        self.ts_handle = term_structure_handle
        self.eval_date = ql.Date(evaluation_date, "%Y-%m-%d")

    def compute_discount_factors_for_grid(
        self, time_grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Extract vectorized discount factors for a given array of year fractions."""
        base_serial = self.eval_date.serialNumber()
        serial_offsets = (time_grid * 365.25).astype(int)

        discount_factors = np.empty_like(time_grid, dtype=np.float64)
        ts = self.ts_handle.currentLink()

        for i, serial in enumerate(base_serial + serial_offsets):
            d = ql.Date(int(serial))
            discount_factors[i] = ts.discount(d)

        return np.ascontiguousarray(discount_factors, dtype=np.float64)

    def price_paths(
        self, cash_flows: npt.NDArray[np.float64], time_grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the Present Value (PV) for each Monte Carlo path simultaneously
        using QuantLib term structure discount factors.
        """
        discount_factors = self.compute_discount_factors_for_grid(time_grid)
        discounted_cash_flows = cash_flows * discount_factors[None, :]
        present_values = np.sum(discounted_cash_flows, axis=1)
        return present_values

    def price_paths_with_liquidity_haircut(
        self,
        cash_flows: npt.NDArray[np.float64],
        time_grid: npt.NDArray[np.float64],
        illiquidity_haircut_rate: float,
    ) -> npt.NDArray[np.float64]:
        """Compute path-wise NPVs incorporating an illiquidity exit haircut (bid-ask / forced sale spread).

        Args:
            cash_flows: NDArray of shape (num_paths, num_steps + 1)
            time_grid: NDArray of time points
            illiquidity_haircut_rate: Proportional reduction factor (e.g., 0.05 for 5% illiquidity discount)
        """
        base_npvs = self.price_paths(cash_flows, time_grid)
        # Vectorized scaling to reflect illiquidity discount
        adjusted_npvs = base_npvs * (1.0 - float(illiquidity_haircut_rate))
        return np.ascontiguousarray(adjusted_npvs, dtype=np.float64)

    def compute_risk_metrics(
        self, present_values: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """Compute core portfolio valuation risk metrics (Expected Value, Value-at-Risk, Expected Shortfall)."""
        mean_val = float(np.mean(present_values))
        std_val = float(np.std(present_values))

        var_95 = float(np.percentile(present_values, 5.0))
        var_99 = float(np.percentile(present_values, 1.0))
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
