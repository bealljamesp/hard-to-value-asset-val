# hard-to-value-asset-val/src/val/models/base.py
"""Base models for the hard-to-value-asset valuation engine."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd


class BaseValuationEngine(ABC):
    """Abstract base class for hard-to-value asset and illiquid portfolio valuation engines.
    Enforces NumPy/Pandas vectorized execution paths without explicit loops.
    """

    def __init__(self, risk_free_rate: float) -> float | int:
        self.risk_free_rate = float(risk_free_rate)

    @abstractmethod
    def compute_discount_factors(
        self, cash_flow_times: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute continuous or discrete discount factors vectorially. O(n) memory allocation."""
        pass

    @abstractmethod
    def evaluate_portfolio(
        self, cash_flows: pd.DataFrame, weights: npt.NDArray[np.float64]
    ) -> dict[str, float | npt.NDArray[np.float64]]:
        """Evaluate portfolio valuation metrics using forced vectorization and native matrix operators (@)."""
        pass
