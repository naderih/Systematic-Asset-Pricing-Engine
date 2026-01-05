# Factor Exposure Construction Module

## Objective
This module constructs the **Time-Varying Factor Exposure Matrix ($X_t$)**, the primary input for the dynamic multifactor risk model.

It implements a professional-grade process to build robust, **Point-in-Time (PIT)** factor exposures. For each rebalance period, it calculates a full cross-section of exposures based strictly on data available at that moment.

The output is a standardized panel where each row is a stock-date tuple, and columns represent sensitivities to fundamental risk drivers.

---

## Methodology

We adhere to the **Grinold & Kahn** framework for robust factor construction, focusing on two key mechanisms: **Composite Factors** and **Capitalization-Weighted Standardization**.

### 1. Industry Classification
Each asset is mapped to one of the **12 Fama-French Industry** groups based on its historical SIC code. These serve as orthogonal dummy variables to capture sector-specific risk variance.

### 2. Composite Style Factors
To reduce the idiosyncratic noise inherent in single metrics (e.g., relying solely on P/E ratio), we construct **Composite Factors** by averaging multiple "descriptors."

* **Value:** Composite of $\ln(B/M)$ and $\ln(E/P)$.
* **Momentum:** Composite of 12-1 Month Returns and 6-1 Month Returns.
* **Size:** $\ln(\text{Market Cap})$.
* **Financial Constraints:** The Whited-Wu Index (a composite of cash flow, leverage, and dividend status).

*> **Note on Lookahead Bias:** All accounting descriptors (Book Equity, Earnings) are lagged appropriately to simulate reporting delays.*

### 3. Cross-Sectional Standardization (The "Barra" Method)
Raw accounting ratios are not comparable across time or assets. We apply **Capitalization-Weighted Standardization** at each time step $t$:

$$
z_{i,t} = \frac{x_{i,t} - \mu_{cap, t}}{\sigma_{cap, t}}
$$

Where $\mu_{cap}$ and $\sigma_{cap}$ are the cap-weighted mean and standard deviation of the cross-section. This ensures that micro-cap outliers do not skew the factor distribution, resulting in a risk model that better reflects the investable universe.

### 4. Re-Standardization
Final composite factors are re-standardized to ensure a pure distribution before entering the risk model:
$$
\text{Mean} \approx 0, \quad \text{Std Dev} \approx 1
$$

---

## Theoretical Justification

### Why Composite Factors?
Financial data is noisy. A firm might look cheap on a P/B basis but expensive on a P/E basis. By averaging these descriptors, we diversify away measurement error and capture the true latent "Value" characteristic more effectively.

### Why Time-Varying Exposures?
Static betas fail to capture firm evolution. A "Growth" stock today may become a "Value" stock in 5 years. By recalculating $X_t$ monthly, the risk model dynamically adapts to the changing nature of the firm.