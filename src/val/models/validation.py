# hard-to-value-asset-val/src/val/models/validation.py
"""Statistical validation and backtesting utilities for hard-to-value asset models."""

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2


class ValuationValidator:
    """Statistical validation and backtesting harness for hard-to-value asset models."""

    @staticmethod
    def kupiec_pof_test(
        failures: int, observations: int, confidence_level: float = 0.95
    ) -> tuple[float, float]:
        """Perform Kupiec's Proportion of Failures (POF) unconditional coverage test.

        Returns:
            test_statistic: Likelihood ratio test statistic
            p_value: Chi-square p-value with 1 degree of freedom
        """
        if observations == 0:
            return 0.0, 1.0

        p = 1.0 - confidence_level
        pi_hat = failures / observations

        # Prevent log(0) errors
        if pi_hat == 0.0:
            lr_stat = -2.0 * np.log((1.0 - p) ** observations)
        elif pi_hat == 1.0:
            lr_stat = -2.0 * np.log(p**observations)
        else:
            term1 = (observations - failures) * np.log(1.0 - p) + failures * np.log(p)
            term2 = (observations - failures) * np.log(
                1.0 - pi_hat
            ) + failures * np.log(pi_hat)
            lr_stat = -2.0 * (term1 - term2)

        p_value = 1.0 - chi2.cdf(lr_stat, df=1)
        return float(lr_stat), float(p_value)

    @staticmethod
    def validate_array_shapes(
        balances: npt.NDArray[np.float64], cash_flows: npt.NDArray[np.float64]
    ) -> bool:
        """Enforces shape invariants and contiguous memory layout constraints."""
        return (
            balances.shape == cash_flows.shape
            and balances.flags["C_CONTIGUOUS"]
            and cash_flows.flags["C_CONTIGUOUS"]
        )
