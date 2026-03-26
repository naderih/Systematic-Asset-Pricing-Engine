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
       We want the Benchmark to have alpha of zero 
       because we seek the component of the signal that is *uncorrelated* with the market.
       We want h_Bᵀ * α_final = 0
       α_final,n = α_raw,n - β_n * α_B_raw
       
       In matrix form: 
       α_final = α_raw - (Beta_(vactor) * α_B_raw_(scalar)

       By making the alpha vector benchmark-neutral, we guarantee that the 
       optimal portfolio constructed from this signal will have zero active beta.

    3. Signal Combination & Smoothing:
       - Combines single factors into a composite (e.g., Long Momentum + Short Constraints).
       - Applies a 3-month rolling average to reduce turnover ("Flicker Reduction").

    4. Standardization:
       Rescales the final signal to unit volatility (Std=1) for the optimizer.
"""

import os
import pandas as pd

class AlphaEngine:
    """
    The engine responsible for turning raw factors into "Pure Alpha".
    
    Responsibilities:
    - Estimating forward-looking market sensitivity (Beta).
    - Neutralizing signals against systematic risk.
    - constructing the final composite signal for the Portfolio Optimizer.
    """
    
    def __init__(self, 
                 panel_path: str,
                 factors_path: str,
                 alpha_output_folder_path: str,  
                 alpha_columns: dict, 
                 beta_window: int = 60, 
                 min_beta_obs:int = 36):
        """
        Args:
            panel_path (str): Path to the panel parquet dataset
            factors_path (str): Path to the factors exposures parquet dataset
            output_fodler_path (str): Path for the saved output alpha_signal parquet file 
            alpha_columns (dict): a dictionary where keys are the name of the alpha columns in the panel data and the values are the sign e.g. {'momentum': 1, 'fin_constraints': -1}, 
            beta_window: The historical window used to estimate the beta, 
            min_beta_obs: The minimum number of observations in beta_window to allow for calculation of beta 
        """
        self.panel_path = panel_path
        self.factors_path = factors_path
        self.alpha_output_folder_path = alpha_output_folder_path
        
        # Configuration for Beta Estimation
        self.beta_window = beta_window        # 5-Year rolling window
        self.min_beta_obs = min_beta_obs       # Minimum 3 years of history required

        self.alpha_columns = alpha_columns

        self.beta_winsor_lower = 0.05 # Lower bound percentile
        self.beta_winsor_upper = 0.95 # Upper bound percentile

    def load_data(self):
        """
        Loads and joins the Panel Data (Returns/Caps) and Factor Exposures (X Matrix).
        """
        print("Loading datasets...")
        
        if not os.path.exists(self.panel_path) or not os.path.exists(self.factors_path):
            raise FileNotFoundError(f"Missing input files in {self.data_dir}")

        panel_data = pd.read_parquet(self.panel_path)
        x_factors = pd.read_parquet(self.factors_path)

        # Ensure multi-index alignment for join
        for df in [panel_data, x_factors]:
            if df.index.names != ['permno', 'date']:
                df.reset_index(inplace=True)
                df.set_index(['permno', 'date'], inplace=True)

        # Merge specific columns required for the alpha process
        market_data = panel_data[['ret_monthly', 'mkt_cap', 'vwretd']]
        alpha_signals = x_factors[[*self.alpha_columns]]
        
        # Inner join to ensure we only process stocks where we have both data types
        df = market_data.join(alpha_signals, how='inner').dropna()
        print(f"Data loaded. Final shape: {df.shape}")
        return df

    def calculate_rolling_betas(self, df): 
        
        """
        Calculates time-varying market betas using rolling OLS.
        
        Model: r_stock = alpha + beta * r_market + epsilon
        Window: 60 months (5 years). calculates if  36 months of data is present
        """
        print("Calculating rolling betas as Cov(r_n, r_mkt) / Var(r_Mkt) ...")
        
        # Pre-calculate Market Variance once for the whole timeline
        # This avoids calculating the same number 1000s of times
        mkt_var = (df.groupby('date')['vwretd'].mean().rolling(window = self.beta_window, 
                                                              min_periods = self.min_beta_obs)
                                                              .var())
        def _rolling_beta_cov(group):
            # Calculate covariance for this specific stock
            cov = (group['ret_monthly'].rolling(window = self.beta_window, 
                                               min_periods = self.min_beta_obs)
                                               .cov(group['vwretd']))
            
            # mkt_var is a series of market variance calculated for every date in our universe 
            # the stock from this specific group may not exist on some of those dates 
            mkt_var_valid = mkt_var.reindex(group.index.get_level_values('date'))
            return cov / mkt_var_valid

        rolling_beta_cov= df.groupby('permno', group_keys = False).apply(_rolling_beta_cov)
        
        df['beta'] = rolling_beta_cov
        
        valid_betas = df['beta'].notna().sum()
        print(f"Beta calculation complete. Valid estimates: {valid_betas} / {len(df)} = {valid_betas/ len(df):.2f}")
        
        return df

       
    def winsorize_betas(self, df):
        """
        Winsorizes betas cross-sectionally (per date) to remove estimation errors.
        
        Rolling regressions on small caps often produce wild beta estimates (e.g., -5.0 or +10.0)
        due to liquidity events or data errors. These outliers would distort the neutralization step.
        """
        print(f"Winsorizing betas ({self.beta_winsor_lower*100}% - {self.beta_winsor_upper*100}%)...")
        
        lower = df.groupby('date')['beta'].transform(lambda x: x.quantile(self.beta_winsor_lower))
        upper = df.groupby('date')['beta'].transform(lambda x: x.quantile(self.beta_winsor_upper))

        df['beta_winsorized'] = df['beta'].clip(lower= lower , upper = upper)
        return df

    def create_composite_and_smooth_alpha(self, df):
        """
        Composite alpha signal is calcualted as the mean of other alpha signals 
        For smoothing, we take a 3-month rolling mean of the raw signals and the composite signal.
        """
        print(f"Constructing composite alpha signal ...")
        running_total = pd.Series(0.0, index = df.index)
        
        for signal, sign in self.alpha_columns.items():
            running_total += sign * df[signal]
        df['composite'] = running_total / len(self.alpha_columns)

        # We take a 3-month rolling mean of the raw composite.
        # This keeps the "trend" but removes the monthly "flicker", reducing trading costs.

        print("Smooothing the alpha signals...")
        signal_cols = [*self.alpha_columns, 'composite']
        for signal in signal_cols:
            smooth = (df.groupby('permno')[signal]
                      .rolling(window = 3, min_periods = 1)
                      .mean()
                      .reset_index(level = 0 , drop = True))
            df[f"{signal}_smooth"] = smooth
        return df 

    def neutralize_signals(self, df):
        """
        Performs Gram- Schmidt orthogonalization for each signal:
        The texhtbook formula is : α_final,n = α_raw,n - β_n * α_B_raw 
        But because we winsorized beta and have missing observations, 
        the formaul is adjusted so that the alpha for the benchmark is forced to zero
        α_final,n = α_raw,n - β_n * α_B_raw / β_market 
        """
        print("Making the alpha signal benchmark-neutral:")
        print("Orthogonalizing Alphas...")

        signal_columns_original = [*self.alpha_columns, 'composite']
        signal_columns_smooth = [c for c in df.columns if c.endswith('smooth')]
        signal_columns = signal_columns_original + signal_columns_smooth


        for signal in signal_columns:
            # 1. Isolate only rows that have complete data for THIS specific signal
            valid = df[['mkt_cap', 'beta_winsorized', signal]].dropna().copy()
            
            #2. calculate cap weights for each date 
            sum_cap = df.groupby(level = 'date')['mkt_cap'].transform('sum')
            weight = valid['mkt_cap'] / sum_cap

            #3. calculate benchmark alpha and beta at each date 
            weight_by_beta = weight * valid['beta_winsorized']
            weight_by_alpha = weight * valid[signal]
            
            valid['bench_beta'] = weight_by_beta.groupby(level = 'date').transform('sum')
            valid['bench_alpha'] = weight_by_alpha.groupby(level = 'date').transform('sum')
            
            #4. calcualte the orthogonalized signal
            signal_orthogonalized = (valid[signal] - valid['beta_winsorized'] 
                                     * (valid['bench_alpha'] / valid['bench_beta']))
            
            
            # Signals were previously standardized in the engine. 
            # However, alpha signals are orthogonalized to the benchmark, 
            # so, they are no longer standardized
            # we dont demean here, because they're orthgonalized with the benchark 
            # and demeaning the signals would violate the benchmakr-neutrality 
            signal_std = signal_orthogonalized.groupby(level = 'date').transform('std')
            df[f'alpha_{signal}'] = signal_orthogonalized / signal_std
            df[f'alpha_{signal}'] = df[f'alpha_{signal}'].fillna(0)

        return df 
    
    def verify_neutrality(self, df):
        alpha_cols = [c for c in df.columns if c.startswith('alpha')]
        market_alpha_means = pd.Series(index = alpha_cols, dtype = float)
        
        sum_cap = df.groupby('date')['mkt_cap'].transform('sum')
        weight = df['mkt_cap'] / sum_cap

        for col in alpha_cols:
            alpha_by_weight = weight * df[col]
            market_alpha_means[col] = alpha_by_weight.groupby(level = 'date').sum().mean()

        max_bias = market_alpha_means.abs().max()
        
        if max_bias < 1e-7:
            print('SUCCESS: All alphas are benchmark-neutral.')
        else:
            print(f'WARNING: Signals retain systematic bias! Max bias: {max_bias}')
            print(market_alpha_means)

    def run_alpha_pipeline(self):
        """
        Executes the full Alpha Generation workflow.
        """
        df = self.load_data()
        df = self.calculate_rolling_betas(df)
        df = self.winsorize_betas(df)
        df = self.create_composite_and_smooth_alpha(df)
        df = self.neutralize_signals(df)
        self.verify_neutrality(df)

        alpha_cols = [c for c in df.columns if c.startswith('alpha')]
        final_signals = df[alpha_cols]

        output_path = os.path.join(self.alpha_output_folder_path, 'alpha_signals.parquet')
        final_signals.to_parquet(output_path)
        print(f"\nPipeline complete. Saved to: {output_path}")

if __name__ == "__main__":
    pass