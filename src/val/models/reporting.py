# hard-to-value-asset-val/src/val/models/reporting.py
"""Reporting utilities for hard-to-value asset portfolio valuations."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


class ValuationReporter:
    """Export and reporting utility for portfolio valuation results and stress grids."""

    @staticmethod
    def generate_summary_report(
        risk_metrics: dict[str, float],
        stress_results: dict[str, npt.NDArray[np.float64]] | None = None,
    ) -> dict[str, object]:
        """Compile core risk metrics and stress-test summaries into a structured report dictionary."""
        report = {
            "engine": "Hard-to-Value Asset Valuation Engine",
            "version": "0.1.0",
            "status": "SUCCESS",
            "risk_metrics": {k: float(v) for k, v in risk_metrics.items()},
        }

        if stress_results is not None:
            report["stress_summary"] = {
                "max_stressed_npv": float(np.max(stress_results["mean_npvs"])),
                "min_stressed_npv": float(np.min(stress_results["mean_npvs"])),
                "worst_case_var_95": float(np.min(stress_results["var_95_grid"])),
            }

        return report

    @staticmethod
    def export_report_json(
        report_dict: dict[str, object], output_path: str | Path
    ) -> None:
        """Export the compiled report dictionary to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4)

    @staticmethod
    def export_stress_grid_csv(
        stress_matrix: npt.NDArray[np.float64], output_path: str | Path
    ) -> None:
        """Export a multi-dimensional stress grid matrix to a CSV file."""
        df = pd.DataFrame(stress_matrix)
        df.to_csv(output_path, index=False)
