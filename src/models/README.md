# Quantitative Core Library

## Overview
This directory houses the mathematical engines of the active management system. It implements the **Grinold-Kahn** framework for decomposing return and risk, separating the investment process into four distinct, modular components:

1.  **Risk Modeling** (`risk.py`): Forecasting the covariance matrix ($V$).
2.  **Alpha Generation** (`alpha.py`): Forecasting residual returns ($\alpha$).
3.  **Portfolio Optimization** (`portfolio.py`): Constructing efficient frontiers.
4.  **Backtesting** (`backtest.py`): Historical validation and attribution.

---

## 1. Dynamic Risk Model (`risk.py`)

### Objective
To forecast the **Active Risk Matrix ($V$)** of the investment universe. We move beyond static historical covariance by decomposing risk into systematic and idiosyncratic components using a **Fundamental Factor Model**:

$$
V_{t+1} = X_t F_{t+1} X_t^T + \Delta_{t+1}
$$

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
Risk clusters in time (e.g., 2008, 2020). A simple average is too slow to react. We use an **Exponentially Weighted Moving Average (EWMA)** with a 36-month half-life to forecast future variance:

1.  **Systematic Risk ($F$):** Computed as the EWMA covariance of the factor return history $f_t$.
2.  **Specific Risk ($\Delta$):** Computed as the EWMA of the squared residuals $u_t^2$.

---

## 2. Alpha Signal Generation (`alpha.py`)

### Objective
To transform raw investment ideas into **"Pure Alpha"** signals. In this framework, a signal is only actionable if it represents a forecast of *residual* return, uncorrelated with the benchmark.

### The Signal Pipeline

1.  **Dynamic Beta Estimation:**
    We estimate the time-varying market beta ($\beta_{i,t}$) of every stock using a **60-month rolling regression**, Winsorized to remove estimation errors from illiquid micro-caps.

2.  **Orthogonalization (Gram-Schmidt):**
    A raw signal (e.g., "Distress") often carries hidden systematic risks. We mathematically strip out these risks to ensure the signal has **Zero Ex-Ante Beta**:
    $$\alpha_{i,t} = S_{i,t} - \left( \beta_{i,t} \times \frac{\text{Cov}(S, \beta)}{\text{Var}(\beta)} \right)$$

3.  **Smoothing & Standardization:**
    * **Turnover Control:** We apply a 3-month rolling mean to the signal to filter out high-frequency noise ("flicker"), reducing trading costs.
    * **Z-Scoring:** The final signal is standardized ($\sigma=1$) to provide a consistent unit of measurement for the optimizer.

---

## 3. Portfolio Optimization (`portfolio.py`)

### Objective
To translate Alpha and Risk forecasts into optimal portfolio weights ($h^*$) that maximize the **Information Ratio** while adhering to strict real-world constraints.

### Methodology

#### The Grinold-Kahn Closed-Form Solution
Instead of a slow quadratic solver, we utilize the analytical solution to the unconstrained Mean-Variance problem:
$$h^* \propto V^{-1} \alpha$$
This elegantly handles diversification: $V^{-1}$ naturally downweights assets that are highly volatile or highly correlated with other holdings.

#### Constraints & Scaling
We apply "Ex-Post" scaling to the analytical solution to enforce mandates:
1.  **Volatility Targeting:** Weights are scaled so the *Ex-Ante Tracking Error* equals the target (e.g., 10%).
2.  **Leverage Limits:** Weights are capped to ensure Gross Exposure $\le$ Max Leverage (e.g., 300%).
3.  **Turnover Smoothing:** We implement an autoregressive filter to dampen trading activity:
    $$h_{final, t} = (1 - \lambda) h_{target, t} + \lambda h_{prev, t}$$
    Where $\lambda$ (e.g., 0.5) controls the "memory" of the portfolio, significantly improving Net Returns by reducing churn.

---

## 4. Backtesting Engine (`backtest.py`)

### Objective
The "Time Machine" that validates the strategy. It replays the historical sequence of optimal portfolios against realized market data.

### Key Features

* **Lag Enforcement:** Strictly aligns Holdings at Time $T$ with Returns at Time $T+1$ to prevent **Lookahead Bias**.
* **Risk-Free Adjustment:** Integrates Fama-French $R_f$ data to calculate accurate Sharpe Ratios.
* **Performance Attribution:** Decomposes returns into:
    * **Benchmark Return:** The passive return from market exposure.
    * **Active Return:** The value added by the strategy ($\alpha$).
    * **Excess Return:** The total return above the risk-free rate.

### Outputs
* **Tearsheet:** Summary metrics (Sharpe, IR, Max Drawdown).
* **Equity Curve:** Full time-series of cumulative wealth.