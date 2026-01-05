"""
Module: Alpha Generation & Neutralization Engine

OBJECTIVE:
    Implements the Grinold-Kahn framework for generating pure, idiosyncratic alpha signals.
    This module transforms raw factor exposures (e.g., Momentum, Financial Constraints)
    into tradeable signals by removing systematic market risk (Beta).

METHODOLOGY:
    1. Dynamic Beta Estimation:
       Calculates time-varying market betas using 60-month rolling regressions 
       (Winsorized to prevent outlier distortion).

    2. Orthogonalization (The "Residual" Approach):
       We seek the component of the signal that is *uncorrelated* with the market.
       Math: Alpha = Signal - (Beta * (Cov(Signal, Beta) / Var(Beta)))
       In a cap-weighted framework, this simplifies to ensuring the signal has 
       zero beta exposure relative to the benchmark.

    3. Signal Combination & Smoothing:
       - Combines single factors into a composite (e.g., Long Momentum + Short Distress).
       - Applies a 3-month rolling average to reduce turnover ("Flicker Reduction").

    4. Standardization:
       Rescales the final signal to unit volatility (Std=1) for the optimizer.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

class AlphaEngine:
    """
    The engine responsible for turning raw factors into "Pure Alpha".
    
    Responsibilities:
    - Estimating forward-looking market sensitivity (Beta).
    - Neutralizing signals against systematic risk.
    - constructing the final composite signal for the Portfolio Optimizer.
    """
    
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to the processed parquet datasets.
        """
        self.data_dir = data_dir
        
        # Configuration for Beta Estimation
        self.beta_window = 60        # 5-Year rolling window
        self.min_beta_obs = 36       # Minimum 3 years of history required
        self.beta_winsor_lower = 0.01 # Lower bound percentile
        self.beta_winsor_upper = 0.99 # Upper bound percentile

    def load_data(self):
        """
        Loads and joins the Panel Data (Returns/Caps) and Factor Exposures (X Matrix).
        """
        print("Loading datasets...")
        panel_path = os.path.join(self.data_dir, 'panel_data.parquet')
        factors_path = os.path.join(self.data_dir, 'factor_exposures.parquet')

        if not os.path.exists(panel_path) or not os.path.exists(factors_path):
            raise FileNotFoundError(f"Missing input files in {self.data_dir}")

        panel_data = pd.read_parquet(panel_path)
        x_factors = pd.read_parquet(factors_path)

        # Ensure multi-index alignment for join
        for df in [panel_data, x_factors]:
            if df.index.names != ['permno', 'date']:
                df.reset_index(inplace=True)
                df.set_index(['permno', 'date'], inplace=True)

        # Merge specific columns required for the alpha process
        market_data = panel_data[['ret_monthly', 'mkt_cap', 'vwretd']]
        alpha_signals = x_factors[['Momentum', 'FinConstraint']]
        
        # Inner join to ensure we only process stocks where we have both data types
        df = market_data.join(alpha_signals, how='inner').dropna()
        print(f"Data loaded. Final shape: {df.shape}")
        return df

    def calculate_rolling_betas(self, df):
        """
        Calculates time-varying market betas using rolling OLS.
        
        Model: r_stock = alpha + beta * r_market + epsilon
        Window: 60 months (5 years).
        """
        print("Calculating rolling betas (this may take a moment)...")

        def _rolling_beta(group):
            if len(group) < self.min_beta_obs:
                return pd.Series(np.nan, index=group.index)
            
            y = group['ret_monthly']
            X = sm.add_constant(group['vwretd'])
            
            try:
                rols = RollingOLS(y, X, window=self.beta_window, min_nobs=self.min_beta_obs)
                results = rols.fit()
                return results.params['vwretd']
            except:
                return pd.Series(np.nan, index=group.index)

        # Group by PERMNO to run regression per stock
        rolling_betas = df.groupby('permno', group_keys=False).apply(_rolling_beta)
        df['beta'] = rolling_betas
        
        # Report coverage statistics
        valid_betas = df['beta'].notna().sum()
        print(f"Beta calculation complete. Valid estimates: {valid_betas} / {len(df)}")
        return df

    def winsorize_betas(self, df):
        """
        Winsorizes betas cross-sectionally (per date) to remove estimation errors.
        
        Why? Rolling regressions on small caps often produce wild beta estimates (e.g., -5.0 or +10.0)
        due to liquidity events or data errors. These outliers would distort the neutralization step.
        """
        print(f"Winsorizing betas ({self.beta_winsor_lower*100}% - {self.beta_winsor_upper*100}%)...")
        
        df['beta_winsorized'] = df.groupby('date')['beta'].transform(
            lambda x: x.clip(
                lower=x.quantile(self.beta_winsor_lower),
                upper=x.quantile(self.beta_winsor_upper)
            )
        )
        return df

    def _neutralize_single_period(self, group, signal_col):
        """
        Internal Helper: Performs Gram-Schmidt orthogonalization for a single date.
        
        Objective: Construct a signal S* such that Beta(S*) = 0.
        Formula:   S* = S - (Beta * k)
        Where k is a scaling factor ensuring the cap-weighted sum of the new signal is zero relative to beta.
        """
        required = ['mkt_cap', 'beta_winsorized', signal_col]
        clean_group = group.dropna(subset=required)

        if clean_group.empty:
            return pd.Series(dtype='float64')

        # 1. Calculate Cap-Weighted properties of the Signal and the Benchmark (Beta)
        weights = clean_group['mkt_cap'] / clean_group['mkt_cap'].sum()
        bench_alpha = (clean_group[signal_col] * weights).sum()
        bench_beta = (clean_group['beta_winsorized'] * weights).sum()

        # 2. Orthogonalize
        # If the benchmark has zero beta (unlikely), we just center the signal.
        if np.isclose(bench_beta, 0):
            return clean_group[signal_col] - bench_alpha
        
        # Calculate the projection of the Signal onto Beta
        adj_factor = bench_alpha / bench_beta
        
        # Subtract the systematic component
        return clean_group[signal_col] - (clean_group['beta_winsorized'] * adj_factor)

    def neutralize_signals(self, df, signal_names=['Momentum', 'FinConstraint']):
        """
        Iterates through signals and removes their systematic market exposure.
        """
        for signal in signal_names:
            print(f"Neutralizing signal: {signal}...")
            # Apply the helper function group-wise by date
            neutralized = df.groupby('date', group_keys=False).apply(
                lambda g: self._neutralize_single_period(g, signal)
            )
            df[f'alpha_{signal}'] = neutralized
        return df

    def finalize_signals(self, df):
        """
        Constructs the final Composite and standardizes it for the optimizer.
        """
        print("Finalizing signals...")
        
        # 1. Create Composite Alpha
        # Logic: We like Momentum (High is Good) and dislike Distress (High is Bad).
        # Therefore: Composite = (Momentum - FinConstraint) / 2
        df['alpha_composite'] = (df['alpha_Momentum'] - df['alpha_FinConstraint']) / 2
        
        # 2. Alpha Smoothing (Turnover Reduction)
        # We take a 3-month rolling mean of the raw composite.
        # This keeps the "trend" but removes the monthly "flicker", significantly reducing trading costs.
        print("Smoothing Alpha (3-month rolling average)...")
        df['alpha_composite'] = df.groupby('permno')['alpha_composite'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        
        # 3. Z-Score Standardization (Unit Variance)
        # We divide by cross-sectional Std Dev so the optimizer sees a signal ~ N(0, 1).
        # Note: We do NOT demean here, because the neutralization step already centered it 
        # relative to the benchmark.
        print("Scaling signals to Unit Variance...")
        targets = {
            'alpha_Momentum': 'alpha_Momentum_final',
            'alpha_FinConstraint': 'alpha_FinConstraint_final',
            'alpha_composite': 'alpha_Composite_final'
        }
        
        for raw_col, final_col in targets.items():
            df[final_col] = df.groupby('date')[raw_col].transform(
                lambda x: x / x.std()
            )
        
        # 4. Cleanup
        final_cols = list(targets.values())
        result_df = df[final_cols].copy()
        result_df.fillna(0, inplace=True)
        
        return result_df

    def verify_neutrality(self, df, final_df):
        """
        Verification Step: Calculates the ex-ante Beta of the final signal.
        If the math is correct, the Cap-Weighted sum of (Signal * Beta) should be effectively zero.
        """
        print("\n--- Verifying Signal Neutrality ---")
        # Re-attach mkt_cap for verification
        check_df = final_df.join(df['mkt_cap'])
        check_df['weight'] = check_df.groupby('date')['mkt_cap'].transform(lambda x: x/x.sum())
        
        # The 'Bias' is the systematic exposure remaining in the portfolio
        bias = (check_df['alpha_Composite_final'] * check_df['weight']).groupby('date').sum().mean()
        
        print(f"Average Benchmark-Weighted Alpha Bias: {bias:.10f}")
        if abs(bias) < 1e-8:
            print("SUCCESS: Signal is orthogonal to the benchmark.")
        else:
            print("WARNING: Signal retains systematic bias.")

    def run_pipeline(self):
        """
        Executes the full Alpha Generation workflow.
        """
        df = self.load_data()
        df = self.calculate_rolling_betas(df)
        df = self.winsorize_betas(df)
        df = self.neutralize_signals(df)
        final_signals = self.finalize_signals(df)
        
        self.verify_neutrality(df, final_signals)
        
        output_path = os.path.join(self.data_dir, 'alpha_signals.parquet')
        final_signals.to_parquet(output_path)
        print(f"\nPipeline complete. Saved to: {output_path}")

if __name__ == "__main__":
    pass