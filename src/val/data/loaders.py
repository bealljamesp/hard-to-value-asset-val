# hard-to-value-asset-val/src/val/data/loaders.py
"""High-performance data loaders for hard-to-value asset cash flows and market rates."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl


class MarketDataLoader:
    """High-performance data loader for hard-to-value asset cash flows, market rates, and portfolio weights."""

    @staticmethod
    def load_cash_flows(
        file_path: str | Path,
    ) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
        """Load asset cash-flow schedules and return both the DataFrame and a C-contiguous NumPy array."""
        df_pl = pl.read_csv(file_path)
        df_pd = df_pl.to_pandas()

        # Extract numerical cash-flow grid (excluding metadata columns if present)
        numeric_cols = df_pd.select_dtypes(include=[np.number]).columns
        cf_array = np.ascontiguousarray(
            df_pd[numeric_cols].to_numpy(dtype=np.float64), dtype=np.float64
        )
        return df_pd, cf_array

    @staticmethod
    def load_market_rates(file_path: str | Path) -> dict[str, list[str | float]]:
        """Load market bootstrap rates (deposits and swaps) from configuration data."""
        df = pd.read_csv(file_path)
        return {
            "deposit_maturities": df[df["instrument"] == "deposit"][
                "maturity"
            ].tolist(),
            "deposit_rates": df[df["instrument"] == "deposit"]["rate"].tolist(),
            "swap_maturities": df[df["instrument"] == "swap"]["maturity"].tolist(),
            "swap_rates": df[df["instrument"] == "swap"]["rate"].tolist(),
        }
