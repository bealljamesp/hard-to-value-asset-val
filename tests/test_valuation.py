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
    num_paths = 50
    num_steps = 5
    cash_flows = np.ones((num_paths, num_steps + 1), dtype=np.float64) * 1000.0
    discount_factors = np.ones(
        (num_steps + 1,), dtype=np.float64
    )  # Flat 1.0 discount for baseline test

    kernel = DCFPricingKernel(discount_factors)
    npvs = kernel.price_paths(cash_flows)

    assert npvs.shape == (50,)
    # 6 time steps * 1000.0 = 6000.0 present value per path under flat 1.0 discount
    np.testing.assert_allclose(npvs, 6000.0)


def test_kupiec_pof_test() -> None:
    # Test unexceptional backtest result
    stat, p_val = ValuationValidator.kupiec_pof_test(
        failures=5, observations=100, confidence_level=0.95
    )
    assert isinstance(stat, float)
    assert isinstance(p_val, float)
    assert p_val >= 0.0
