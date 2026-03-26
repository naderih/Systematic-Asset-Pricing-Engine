# Quantitative Core Library

## Overview
This directory houses the mathematical engines of the active management system. It implements the **Grinold-Kahn** framework for decomposing return and risk, separating the investment process into four distinct, modular components:

1.  **Risk Modeling** (`risk.py`): Forecasting the covariance matrix ($V$).
2.  **Alpha Generation** (`alpha.py`): Forecasting residual returns ($\alpha$).
3.  **Portfolio Optimization** (`portfolio.py`): Constructing efficient frontiers via a dual-solver architecture.
4.  **Backtesting** (`backtest.py`): Historical validation, friction modeling, and attribution.

---

## 1. Dynamic Risk Model (`risk.py`)

### Objective
To forecast the **Active Risk Matrix ($V$)** of the investment universe. We move beyond static historical covariance by decomposing risk into systematic and idiosyncratic components using a **Fundamental Factor Model**:

$$V_{t+1} = X_t F_{t+1} X_t^T + \Delta_{t+1}$$

* **$X$**: Factor Exposures (Size, Momentum, Industry).
* **$F$**: Factor Covariance Matrix (Systematic Risk).
* **$\Delta$**: Idiosyncratic Variance Matrix (Specific Risk).

### Methodology

#### Phase 1: Historical Estimation (Fama-MacBeth)
We isolate the pure returns of each factor by running a cross-sectional regression at every time step $t$:

$$r_{i,t} = \sum_{k=1}^{K} X_{i,k,t} f_{k,t} + u_{i,t}$$

This generates two time-series:
* **Factor Returns ($f_t$):** The premium earned by a unit exposure to factor $k$.
* **Specific Returns ($u_t$):** The residual return of stock $i$ unexplained by the model.

#### Phase 2: Volatility Forecasting (EWMA)
Risk clusters in time (e.g., 2008, 2020). A simple average is too slow to react. We use an **Exponentially Weighted Moving Average (EWMA)** with a customizable half-life (e.g., 36 months) to forecast future variance:

1.  **Systematic Risk ($F$):** Computed as the EWMA covariance of the factor return history $f_t$.
2.  **Specific Risk ($\Delta$):** Computed as the EWMA of the squared residuals $u_t^2$.

---

## 2. Alpha Signal Generation (`alpha.py`)

### Objective
To transform raw investment ideas into **"Pure Alpha"** signals. In this framework, a signal is only actionable if it represents a forecast of *residual* return, uncorrelated with the benchmark.

### The Signal Pipeline

1.  **Dynamic Beta Estimation:**
    We estimate the time-varying market beta ($\beta_{i,t}$) of every stock using a rolling regression window (e.g., 60 months), requiring a minimum history (e.g., 36 months) to ensure stability.

2.  **Orthogonalization (Gram-Schmidt):**
    A raw signal often carries hidden systematic risks. We mathematically strip out these risks to ensure the signal has **Zero Ex-Ante Beta**:

    $$\alpha_{i,t} = S_{i,t} - \left( \beta_{i,t} \times \frac{\text{Cov}(S, \beta)}{\text{Var}(\beta)} \right)$$

3.  **Smoothing & Standardization:**
    * **Turnover Control:** We apply a rolling mean to the signal to filter out high-frequency noise ("flicker"), reducing trading costs.
    * **Z-Scoring:** The final signal is cross-sectionally standardized ($\sigma=1$) to provide a consistent unit of measurement for the optimizer.

---

## 3. Portfolio Optimization (`portfolio.py`)

### Objective
To translate Alpha and Risk forecasts into optimal portfolio weights ($h^*$) that maximize the **Information Ratio** while adhering to strict real-world constraints.

### Methodology
The engine features a dual-solver architecture to transition from theoretical research to tradable reality.

#### 1. The Analytical Solver (Unconstrained Mapping)
For rapid frontier mapping, we utilize the analytical solution to the unconstrained Mean-Variance problem:

$$h^* \propto V^{-1} \alpha$$

This naturally downweights assets that are highly volatile or highly correlated. We then apply **Ex-Post Scaling** for Volatility Targeting and Leverage Limits, alongside an autoregressive smoothing filter:

$$h_{final, t} = (1 - \lambda) h_{target, t} + \lambda h_{prev, t}$$

#### 2. The CVXPY Quadratic Programmer (Institutional Constraints)
For production-grade portfolio construction, we bypass the analytical solution and deploy a convex optimizer to enforce strict institutional mandates:
* **Dollar-Neutrality:** Forces $\sum h_i = 0$ to strictly eliminate passive market exposure.
* **Hard Turnover Ceilings:** Constrains the $L_1$ norm of required trades ($\sum |h_t - h_{drifted, t-1}| \le \text{Max Turnover} \times 2$) to enforce absolute capacity limits.
* **Leverage Bounds:** Caps the $L_1$ norm of total gross weights.

---

## 4. Backtesting Engine (`backtest.py`)

### Objective
The "Time Machine" that validates the strategy. It replays the historical sequence of optimal portfolios against realized market data, heavily prioritizing the modeling of real-world frictions.

### Key Features

* **Lag Enforcement:** Strictly aligns Holdings at Time $T$ with Returns at Time $T+1$ to prevent **Lookahead Bias**.
* **Passive Weight Drift:** Mathematically accounts for the intra-month drift of portfolio weights caused by asset price movements before calculating end-of-month turnover.
* **Transaction Costs (Net of Fees):** Calculates exact one-sided turnover and applies a deterministic penalty (e.g., 10 bps) to the subsequent month's capital base, producing a rigorous net-of-fee performance record.
* **Smart Caching:** Interrogates existing configuration hashes to skip redundant CVXPY parameter sweeps, optimizing compute time.

### Outputs
* **Institutional Tearsheet:** Comprehensive summary metrics (CAGR, Net Sharpe, Information Ratio, Max Drawdown, Average Turnover).
* **Equity Curve:** Full time-series of net cumulative wealth for visualization and comparative drawdown analysis.