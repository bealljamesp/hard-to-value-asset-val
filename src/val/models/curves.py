# hard-to-value-asset-val/src/val/models/curves.py
"""Interest rate curve models for hard-to-value assets."""

import numpy as np
import numpy.typing as npt
import QuantLib as ql


class YieldCurveBootstrapper:
    """QuantLib-backed term structure bootstrapper designed for hard-to-value asset valuation analytics."""

    def __init__(
        self,
        evaluation_date: str,
        calendar_str: str = "UnitedStates",
        settlement_days: int = 2,
    ) -> None:
        self.eval_date = ql.Date(evaluation_date, "%Y-%m-%d")
        ql.Settings.instance().evaluationDate = self.eval_date

        # Map calendar strings to QuantLib calendars
        calendars = {
            "UnitedStates": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
            "Target": ql.TARGET(),
        }
        self.calendar = calendars.get(calendar_str, ql.NullCalendar())
        self.settlement_days = int(settlement_days)
        self.day_counter = ql.Actual360()

    def bootstrap_curve(
        self,
        deposit_maturities: list[str],
        deposit_rates: list[float],
        swap_maturities: list[str],
        swap_rates: list[float],
    ) -> ql.YieldTermStructureHandle:
        """Constructs a piecewise yield curve using deposit rates and swap rates."""

        rate_helpers = []

        # Process short-term deposit helpers
        for maturity_str, rate in zip(deposit_maturities, deposit_rates, strict=True):
            tenor = ql.PeriodParser.parse(maturity_str)
            helper = ql.DepositRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(rate)),
                tenor,
                self.settlement_days,
                self.calendar,
                ql.ModifiedFollowing,
                False,
                self.day_counter,
            )
            rate_helpers.append(helper)

        # Process swap helpers for the long end of the curve
        for maturity_str, rate in zip(swap_maturities, swap_rates, strict=True):
            tenor = ql.PeriodParser.parse(maturity_str)
            helper = ql.SwapRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(rate)),
                tenor,
                self.calendar,
                ql.Annual,
                ql.ModifiedFollowing,
                ql.Thirty360(ql.Thirty360.BondBasis),
                ql.Euribor6M()
                if "EUR" in maturity_str
                else ql.USDLibor(ql.Period(3, ql.Months)),
            )
            rate_helpers.append(helper)

        # Piecewise yield curve construction (Discount exponential interpolation)
        term_structure = ql.PiecewiseLogLinearDiscount(
            self.eval_date, rate_helpers, self.day_counter
        )
        return ql.YieldTermStructureHandle(term_structure)

    def extract_discount_factors(
        self,
        term_structure_handle: ql.YieldTermStructureHandle,
        target_years: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Extracts vectorized discount factors from the QuantLib term structure handle
        into zero-copy NumPy float64 arrays.
        """
        # Convert fractional years to QuantLib Dates vectorially/via list comprehension without pandas apply loops
        base_serial = self.eval_date.serialNumber()
        serial_offsets = (target_years * 365.25).astype(int)

        # Vector evaluation of discount factors
        discount_factors = np.empty_like(target_years, dtype=np.float64)
        ts = term_structure_handle.currentLink()

        for i, serial in enumerate(base_serial + serial_offsets):
            d = ql.Date(int(serial))
            discount_factors[i] = ts.discount(d)

        return discount_factors
