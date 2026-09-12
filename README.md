[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pytest](https://img.shields.io/badge/pytest-100%25-green.svg)](https://docs.pytest.org/)

# Hard-to-Value Asset Valuation & Risk Engine

A high-performance quantitative valuation and risk engine designed for illiquid assets, private credit portfolios, and hard-to-value structured notes. The architecture enforces zero-copy C-contiguous memory layouts, strict vectorization (avoiding explicit Python loops and `.apply()` bottlenecks), and institutional-grade QuantLib yield curve bootstrapping.

---

## Key Features

- **QuantLib Term Structure Integration:** Piecewise log-linear discount curve bootstrapping from market deposit and swap rates.
- **Vectorized Monte Carlo Simulation:** Stochastic cash-flow generation using Geometric Brownian Motion with jump-to-default intensity over C-contiguous NumPy arrays.
- **Liquidity-Adjusted Pricing Kernel:** Present value discounting with integrated secondary-market exit haircuts and bid-ask penalty matrices.
- **Macroeconomic Stress-Testing Harness:** Multi-dimensional grid evaluation measuring portfolio Value-at-Risk (VaR) and Expected Shortfall under severe default intensity and volatility shocks.
- **Statistical Backtesting:** Kupiec's Proportion of Failures (POF) unconditional coverage test suite.
- **Modern Python Architecture:** Built for Python 3.12+ leveraging native generics (PEP 585/604) and automated CI/CD via GitHub Actions.

---

## Project Structure

```text
hard-to-value-asset-val/
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated GitHub Actions test pipeline
├── src/
│   └── val/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── curves.py       # QuantLib yield curve bootstrapper
│       │   └── loaders.py      # Polars/Pandas high-performance data loaders
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cashflows.py    # Vectorized cash-flow path simulator
│       │   ├── pricing.py      # DCF pricing kernel & liquidity adjustments
│       │   ├── reporting.py    # JSON and CSV report compilation engines
│       │   ├── stress.py       # Macroeconomic stress-testing grid
│       │   └── validation.py   # Kupiec backtesting and shape validators
│       └── main.py             # End-to-end execution pipeline
├── tests/
│   ├── __init__.py
│   └── test_valuation.py       # Comprehensive pytest validation suite
├── pyproject.toml              # Modern build system & tool configuration
└── README.md
```

---

## Install and Setup

```
git clone [https://github.com/YOUR_USERNAME/hard-to-value-asset-val.git](https://github.com/YOUR_USERNAME/hard-to-value-asset-val.git)
cd hard-to-value-asset-val
conda create -n hard-val python=3.12 -c conda-forge
conda activate hard-val
pip install -e .[dev]
```

---

## Run the Pipeline and Test Suite

```
python src/val/main.py

pytest -v
```

---

## Notes

- Zero-Copy Memory Discipline: Array mappings explicitly enforce C-contiguous layouts (order='C', np.ascontiguousarray), avoiding memory fragmentation during linear algebra operations.

- QuantLib & Polars Integration: High-performance ingestion via Polars/Arrow combined with precise QuantLib yield curve bootstrapping provides real institutional-grade fluency.

- Robust CI Pipeline: The GitHub Actions workflow cleanly targets Python 3.12 and installs the package in editable development mode (.[dev]), ensuring automated validation on every commit.

---
