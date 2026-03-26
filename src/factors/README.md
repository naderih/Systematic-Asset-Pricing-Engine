# Factor Exposure Construction Module (`engine.py`)

## Objective
This module constructs the **Time-Varying Factor Exposure Matrix ($X_t$)**, the foundational input for both the dynamic multifactor risk model and the alpha generation engine.

It implements a production-grade process to build robust, **Point-in-Time (PIT)** factor exposures. For each rebalance period, it calculates a full cross-section of exposures based strictly on data available at that exact moment, writing the final output to a high-speed `.parquet` panel. 

The output is a standardized matrix where each row is a stock-date tuple, and columns represent sensitivities to fundamental risk drivers and proprietary alpha signals.

---

## Methodology

We adhere to the **Grinold & Kahn** framework for robust factor construction, focusing on Composite Factors, robust outlier handling (Winsorization), and Cross-Sectional Standardization.

### 1. Industry Classification
Each asset is mapped to one of the **12 Fama-French Industry** groups based on its historical SIC code. These serve as orthogonal dummy variables to capture sector-specific risk variance and ensure sector-neutrality where required.

### 2. Composite Style Factors
To reduce the idiosyncratic noise inherent in single metrics (e.g., relying solely on the P/E ratio), we construct **Composite Factors** by aggregating multiple fundamental "descriptors."

* **Value:** Composite of $\ln(B/M)$ and $\ln(E/P)$.
* **Momentum:** Composite of 12-1 Month Returns and 6-1 Month Returns.
* **Size:** $\ln(\text{Market Cap})$.
* **Financial Constraints:** The Whited-Wu Index (a proprietary composite of cash flow, leverage, and dividend status representing the core alpha edge).

*> **Note on Lookahead Bias:** All accounting descriptors (Book Equity, Earnings) are strictly lagged to simulate realistic SEC reporting delays.*

### 3. Outlier Winsorization & Cross-Sectional Standardization (The "Barra" Method)
Raw accounting ratios are not comparable across time or assets, and financial data is notoriously prone to extreme outliers. 

First, all raw descriptors undergo cross-sectional winsorization to cap extreme deviations. Next, we apply **Cross-Sectional Standardization** at each time step $t$:

$$z_{i,t} = \frac{x_{i,t} - \mu_{cap, t}}{\sigma_{cap, t}}$$

Where $\mu_{cap}$ and $\sigma_{cap}$ are the cap-weighted mean and standard deviation of the cross-section. This ensures that micro-cap outliers do not skew the factor distribution, resulting in a risk matrix that accurately reflects the investable institutional universe.

### 4. Re-Standardization
Final composite factors are re-standardized to ensure a pure normal distribution before entering the risk model and optimizer:

$$\text{Mean} \approx 0, \quad \text{Std Dev} \approx 1$$

---

## Theoretical Justification

### Why Composite Factors?
Financial data is noisy. A firm might look cheap on a P/B basis but expensive on a P/E basis. By averaging these descriptors, we diversify away measurement error and capture the true latent characteristic (e.g., "Value") much more effectively than any single ratio.

### Why Time-Varying Exposures?
Static betas fail to capture firm evolution. A "Growth" stock today may become a "Value" stock in 5 years, or a firm's Financial Constraint profile may deteriorate rapidly during a credit crunch. By recalculating $X_t$ monthly, the risk model dynamically adapts to the changing nature of the firm, ensuring the optimizer is always acting on the most current fundamental reality.