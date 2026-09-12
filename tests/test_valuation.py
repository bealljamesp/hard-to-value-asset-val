# tests/test_valuation.py
"""Unit tests for hard-to-value asset valuation models."""

import numpy as np

from val.models.cashflows import IlliquidCashFlowSimulator
from val.models.pricing import DCFPricingKernel
from val.models.validation import ValuationValidator


def test_simulator_shape_and_continuity() -> None:
    simulator = IlliquidCashFlowSimulator(
        initial_balance=1_000_000.0,
        coupon_rate=0.06,
        default_intensity=0.03,
        volatility=0.15,
        time_horizon=1.0,
        num_steps=12,
        num_paths=100,
        seed=42,
    )
    balances, cash_flows = simulator.simulate_paths()

    # Verify dimensions: (num_paths, num_steps + 1)
    assert balances.shape == (100, 13)
    assert cash_flows.shape == (100, 13)

    # Verify C-contiguous memory flag for SIMD optimization
    assert ValuationValidator.validate_array_shapes(balances, cash_flows)


def test_dcf_pricing_kernel() -> None:
    bootstrapper = YieldCurveBootstrapper("2026-09-12")
    ts_handle = bootstrapper.bootstrap_curve(
        deposit_maturities=["1M"],
        deposit_rates=[0.05],
        swap_maturities=["1Y"],
        swap_rates=[0.055],
    )

    kernel = DCFPricingKernel(ts_handle, "2026-09-12")
    time_grid = np.linspace(0.0, 1.0, 6)
    cash_flows = np.ones((50, 6), dtype=np.float64) * 1000.0

    npvs = kernel.price_paths(cash_flows, time_grid)
    assert npvs.shape == (50,)
    assert np.all(npvs > 0.0)


def test_kupiec_pof_test() -> None:
    # Test unexceptional backtest result
    stat, p_val = ValuationValidator.kupiec_pof_test(
        failures=5, observations=100, confidence_level=0.95
    )
    assert isinstance(stat, float)
    assert isinstance(p_val, float)
    assert p_val >= 0.0


from val.data.curves import YieldCurveBootstrapper
from val.models.stress import PortfolioStressTester


def test_quantlib_curve_and_pricing_integration() -> None:
    bootstrapper = YieldCurveBootstrapper("2026-09-12")
    # Bootstrap simple curve
    ts_handle = bootstrapper.bootstrap_curve(
        deposit_maturities=["1M", "3M"],
        deposit_rates=[0.05, 0.052],
        swap_maturities=["1Y", "2Y"],
        swap_rates=[0.055, 0.058],
    )

    kernel = DCFPricingKernel(ts_handle, "2026-09-12")
    time_grid = np.linspace(0.0, 1.0, 6)
    cash_flows = np.ones((10, 6), dtype=np.float64) * 100.0

    npvs = kernel.price_paths(cash_flows, time_grid)
    assert npvs.shape == (10,)
    assert np.all(npvs > 0.0)


def test_stress_tester_grid() -> None:
    bootstrapper = YieldCurveBootstrapper("2026-09-12")
    ts_handle = bootstrapper.bootstrap_curve(
        deposit_maturities=["1M"],
        deposit_rates=[0.05],
        swap_maturities=["1Y"],
        swap_rates=[0.055],
    )
    kernel = DCFPricingKernel(ts_handle, "2026-09-12")
    time_grid = np.linspace(0.0, 1.0, 4)

    tester = PortfolioStressTester(
        initial_balance=500_000.0,
        coupon_rate=0.07,
        time_horizon=1.0,
        num_steps=3,
        num_paths=50,
    )

    di_shocks = np.array([0.02, 0.05], dtype=np.float64)
    vol_shocks = np.array([0.10, 0.20], dtype=np.float64)

    results = tester.run_stress_grid(di_shocks, vol_shocks, kernel, time_grid)
    assert results["mean_npvs"].shape == (2, 2)
    assert results["var_95_grid"].shape == (2, 2)
