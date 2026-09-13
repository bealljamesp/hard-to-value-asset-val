# hard-to-value-asset-val/tests/test_valuation.py
"""Unit tests for hard-to-value asset valuation models."""

import numpy as np
import pandas as pd

from val.data.curves import YieldCurveBootstrapper
from val.data.loaders import MarketDataLoader
from val.models.cashflows import IlliquidCashFlowSimulator
from val.models.pricing import DCFPricingKernel
from val.models.reporting import ValuationReporter
from val.models.stress import PortfolioStressTester
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
    assert balances.shape == (100, 13)
    assert cash_flows.shape == (100, 13)
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
    stat, p_val = ValuationValidator.kupiec_pof_test(
        failures=5, observations=100, confidence_level=0.95
    )
    assert isinstance(stat, float)
    assert isinstance(p_val, float)
    assert p_val >= 0.0


def test_market_data_loader(tmp_path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    p = d / "sample_cf.csv"

    # Create sample CSV file
    df_sample = pd.DataFrame(
        {"t0": [1000.0, 2000.0], "t1": [1100.0, 2100.0], "t2": [1200.0, 2200.0]}
    )
    df_sample.to_csv(p, index=False)

    df_res, arr_res = MarketDataLoader.load_cash_flows(p)
    assert isinstance(df_res, pd.DataFrame)
    assert arr_res.shape == (2, 3)
    assert arr_res.flags["C_CONTIGUOUS"]


def test_valuation_reporter(tmp_path) -> None:
    metrics = {"mean_npv": 250_000.0, "var_95": 100_000.0}
    report = ValuationReporter.generate_summary_report(metrics)

    assert report["status"] == "SUCCESS"
    assert report["risk_metrics"]["mean_npv"] == 250_000.0

    json_path = tmp_path / "report.json"
    ValuationReporter.export_report_json(report, json_path)
    assert json_path.exists()


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
