# hard-to-value-asset-val/src/val/main.py
"""Entry point for the Hard-to-Value Asset Valuation Engine."""

import numpy as np

from val.data.curves import YieldCurveBootstrapper
from val.models.cashflows import IlliquidCashFlowSimulator
from val.models.pricing import DCFPricingKernel
from val.models.reporting import ValuationReporter
from val.models.stress import PortfolioStressTester


def main() -> None:
    print("Initializing Hard-to-Value Asset Valuation Engine...")
    eval_date = "2026-09-12"

    # 1. Bootstrap QuantLib Yield Curve
    print("Bootstrapping term structure...")
    bootstrapper = YieldCurveBootstrapper(eval_date)
    ts_handle = bootstrapper.bootstrap_curve(
        deposit_maturities=["1M", "3M"],
        deposit_rates=[0.05, 0.052],
        swap_maturities=["1Y", "2Y"],
        swap_rates=[0.055, 0.058],
    )

    # 2. Simulate Illiquid Asset Cash Flows
    print("Executing stochastic cash-flow simulation paths...")
    time_horizon = 1.0
    num_steps = 12
    simulator = IlliquidCashFlowSimulator(
        initial_balance=1_000_000.0,
        coupon_rate=0.065,
        default_intensity=0.03,
        volatility=0.18,
        time_horizon=time_horizon,
        num_steps=num_steps,
        num_paths=1000,
        seed=42,
    )
    _, cash_flows = simulator.simulate_paths()
    time_grid = np.linspace(0.0, time_horizon, num_steps + 1)

    # 3. Price Paths via DCF Kernel
    print("Pricing paths via QuantLib term structure integration...")
    pricing_kernel = DCFPricingKernel(ts_handle, eval_date)
    npvs = pricing_kernel.price_paths(cash_flows, time_grid)
    risk_metrics = pricing_kernel.compute_risk_metrics(npvs)
    print(f"Baseline Mean NPV: ${risk_metrics['mean_npv']:,.2f}")
    print(f"Baseline 95% VaR:  ${risk_metrics['var_95']:,.2f}")

    # 4. Run Macroeconomic Stress Testing Grid
    print("Running stress-testing grid...")
    stress_tester = PortfolioStressTester(
        initial_balance=1_000_000.0,
        coupon_rate=0.065,
        time_horizon=time_horizon,
        num_steps=num_steps,
        num_paths=200,
    )
    di_shocks = np.array([0.02, 0.04, 0.06], dtype=np.float64)
    vol_shocks = np.array([0.10, 0.20, 0.30], dtype=np.float64)
    stress_results = stress_tester.run_stress_grid(
        di_shocks, vol_shocks, pricing_kernel, time_grid
    )

    # 5. Compile and Export Reports
    print("Compiling and exporting valuation report...")
    report = ValuationReporter.generate_summary_report(risk_metrics, stress_results)
    ValuationReporter.export_report_json(report, "valuation_report.json")
    ValuationReporter.export_stress_grid_csv(
        stress_results["mean_npvs"], "stress_grid_npv.csv"
    )
    print(
        "Pipeline execution complete. Outputs saved to valuation_report.json and stress_grid_npv.csv."
    )


if __name__ == "__main__":
    main()
