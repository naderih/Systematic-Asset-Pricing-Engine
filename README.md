# Systematic Asset Pricing Engine

## Overview
This repository implements an end-to-end active portfolio management system based on the **Grinold-Kahn framework**. It is designed to rigorously test quantitative investment strategies by isolating idiosyncratic alpha from systematic risk.

The system features a modular architecture with dedicated engines for risk modeling, signal generation, portfolio optimization, and historical backtesting. It prioritizes data integrity, strictly enforcing point-in-time logic to eliminate lookahead bias.

## Key Features
* **Dynamic Risk Modeling:** Implements a Fundamental Factor Risk Model using Fama-MacBeth regressions and Recursive EWMA forecasting.
* **Alpha Purification:** Uses Gram-Schmidt Orthogonalization to neutralize systematic factor exposure, ensuring signals represent pure residual returns.
* **Convex Optimization:** Solves unconstrained Mean-Variance problems with ex-post constraints for leverage and volatility targeting.
* **Transaction Cost Control:** Implements holdings smoothing to reduce turnover and improve net-of-fee performance.

## Repository Structure

```text
.
├── main.ipynb                   # Orchestration Dashboard (Run this first)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git configuration
└── src/                         # Core Source Code
    └── models/
        ├── risk.py              # Covariance forecasting engine
        ├── alpha.py             # Signal generation and neutralization
        ├── portfolio.py         # Mean-Variance optimizer
        └── backtest.py          # Historical simulation engine
```

## Installation

1.  **Clone the repository**
    git clone https://github.com/naderih/Systematic-Asset-Pricing-Engine.git
    cd Systematic-Asset-Pricing-Engine

2.  **Install dependencies**
    pip install -r requirements.txt

## Usage

The entire pipeline is orchestrated via the `main.ipynb` notebook. It executes the following workflow in sequence:

1.  **Risk Model Estimation:** Loads panel data and estimates the $V$ matrix (Factor Covariance + Specific Risk).
2.  **Alpha Construction:** Generates raw signals (e.g., Financial Constraints), neutralizes them against market beta, and standardizes scores.
3.  **Portfolio Optimization:** Generates optimal weights $h^*$ targeting a specific Information Ratio and Tracking Error.
4.  **Backtesting:** Simulates performance from 1995–Present, calculating Sharpe Ratios, Information Ratios, and Drawdowns.

## Methodological Framework

### 1. Risk Modeling
We decompose asset risk into systematic and specific components using a structured factor model:
$V_t = X_t F_t X_t^T + \Delta_t$

* **Systematic Risk ($F$):** Forecasted using a Recursive EWMA of factor returns derived from cross-sectional Fama-MacBeth regressions.
* **Specific Risk ($\Delta$):** Forecasted using a Recursive EWMA of squared regression residuals.

### 2. Alpha Generation
We isolate the "Quality" premium by targeting Financial Constraints (Whited-Wu Index). To ensure the signal provides distinct information from the market, we enforce orthogonality:
$h_B^T \alpha = 0$

This is achieved via beta-adjusted neutralization, effectively stripping out the market component from the raw signal.

### 3. Portfolio Construction
The portfolio optimizer solves for the vector of active weights $h_{PA}$ that maximizes the expected Information Ratio:
$h_{PA}^* \propto V^{-1} \alpha$

We apply a holdings smoothing filter ($\lambda=0.5$) to dampen rebalancing noise, significantly reducing turnover with minimal impact on signal fidelity.

## Performance Summary
* **Signal:** Financial Constraints (Distress Factor)
* **Period:** 1995 – 2024
* **Information Ratio:** ~0.32
* **Annualized Active Return:** ~2.5%
* **Avg Turnover:** ~85%

## Future Roadmap
* **LLM-Based Financial Constraints:** Currently, the system relies on the accounting-based Whited-Wu index to measure distress. A planned enhancement is to utilize Large Language Models (LLMs) to parse 10-K/10-Q filings and earnings call transcripts. By quantifying sentiment and specific language related to credit access and liquidity, we aim to construct a more timely and nuanced "Text-Based Financial Constraint" signal that reacts faster than quarterly accounting updates.

*Past performance is simulated and does not guarantee future results.*