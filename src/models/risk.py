"""
Module: Dynamic Risk Model Engine

OBJECTIVE:
    Constructs a Fundamental Factor Risk Model using the Fama-MacBeth regression approach.
    
    The goal is to decompose asset returns into:
    1. Systematic Risk (Driven by Factors like Size, Momentum, Industry).
    2. Idiosyncratic Risk (Stock-specific noise).

METHODOLOGY:
    1. Cross-Sectional Regression (Fama-MacBeth Step 1):
       For each time period t, we regress the cross-section of stock returns (R_t) 
       on their factor exposures (X_t) using OLS or WLS with specific risk as regression weights:
       R_{i,t} = X_{i,t} * F_t + u_{i,t}
       
       This gives us:
       - F_t: The realized return of each factor for that month.
       - u_{i,t}: The specific (residual) return of each stock.

    2. Risk Forecasting (EWMA):
       We use the time-series of F_t and u_{i,t} to forecast the *future* risk matrix.
       - Factor Covariance (F): EWMA covariance of F_t.
       - Idiosyncratic Variance (Delta): EWMA of squared residuals (u^2).

OUTPUT:
    Saves 'factor_cov_matrices.parquet' and 'idio_vol.parquet' for the Portfolio Optimizer.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

class RiskModel:
    """
    The engine responsible for estimating and forecasting the Risk Matrix (V).
    
    Attributes:
        half_life (int): The decay parameter for the EWMA forecast (in months).
                         36 months is standard for a medium-term risk model.
    """
    
    def __init__(self, 
                 panel_path, 
                 factors_path, 
                 risk_output_folder_path,
                 half_life=36, 
                 regression_weighting = "OLS"):
        
        self.panel_path = panel_path
        self.factors_path = factors_path
        self.risk_output_folder_path = risk_output_folder_path
        self.half_life = half_life

        if regression_weighting not in ['OLS', 'WLS']:
            raise ValueError("regression_method must be 'OLS' or 'WLS'")
        self.regression_weighting = regression_weighting
        
    def load_data(self):
        """
        Loads the two required inputs:
        1. Panel Data (contains 'ret_monthly' - the Dependent Variable Y)
        2. Factor Exposures (contains Style/Industry factors - the Regressors X)
        """
        print("Loading Data for Risk Model...")
        
        # Load Returns
        if not os.path.exists(self.panel_path):
            raise FileNotFoundError(f"Missing {self.panel_path}")
        self.panel_data = pd.read_parquet(self.panel_path)
        
        # Load Exposures (The X matrix)
        if not os.path.exists(self.factors_path):
            raise FileNotFoundError(f"Missing {self.factors_path}")
        self.exposures = pd.read_parquet(self.factors_path)
        
        # --- Robust Index Handling ---
        # Ensure we have a clean (permno, date) MultiIndex for alignment
        self._ensure_multiindex(self.panel_data)
        self._ensure_multiindex(self.exposures)

        print("Data Loaded.")

    def _ensure_multiindex(self, df):
        """Helper to standardize index structure."""
        if df.index.names != ['permno', 'date']:
            df.reset_index(inplace=True)
            df.set_index(['permno', 'date'], inplace=True)
        df.sort_index(inplace=True)

    def run_fama_macbeth(self):
        """
        Runs period-by-period cross-sectional regressions to extract 
        Factor Returns (F_t) and Specific Returns (u_t).
        """
        print("Running Fama-MacBeth Regressions to extract Factor Returns (F_t) and Specific Returns (u_t)")
        print(f"using cross-sectional {self.regression_weighting} regressions ...")
        
        # 1. Align Data
        # We perform an INNER JOIN on (permno, date). 
        # A stock must have both a return and exposures to be included in the risk model estimation.
        y = self.panel_data[['ret_monthly', 'mkt_cap']]
        X = self.exposures
        
        aligned = y.join(X, how='inner')
        
        # Sanity Check
        if aligned.empty:
            print("ERROR: Intersection of Panel Data and Exposures is EMPTY.")
            print(f"Panel Sample: {y.index[:2]}")
            print(f"Exposures Sample: {X.index[:2]}")
            raise ValueError("Data Alignment Failed: No common (permno, date) rows found.")
            
        print(f"Aligned Data: {len(aligned)} rows found across {aligned.index.get_level_values('date').nunique()} dates.")
        
        # 2. Iterate by Date
        dates = aligned.index.get_level_values('date').unique().sort_values()
        
        factor_ret_list = []
        resid_list = []
        
        # Identify regressors (everything except the return column)
        factor_cols = [c for c in aligned.columns if c not in ['ret_monthly', 'mkt_cap']]
        
        for date in dates:
            # Efficiently slice data for the current month
            # .xs(date) drops the date level, leaving 'permno' as the index
            try:
                monthly_slice = aligned.xs(date, level='date')
            except KeyError:
                continue
                
            y_t = monthly_slice['ret_monthly']
            X_t = monthly_slice[factor_cols].astype(float)

            # Constraint: Need more observations than factors to solve OLS
            if len(y_t) < len(factor_cols) + 2:
                continue

            try:
                if self.regression_weighting == "OLS":
                # OLS Regression: R_i = Beta * F + epsilon
                # Note: We do NOT add a constant because 'Market' or 'Industry' factors usually span the intercept.
                    model = sm.OLS(y_t, X_t, missing='drop')
                    results = model.fit()
                elif self.regression_weighting == "WLS":
                    weights = np.sqrt(monthly_slice['mkt_cap'].astype(float))
                    results = sm.WLS(y_t, X_t, weights=weights, missing='drop').fit()

                # A. Store Factor Returns (Coefficients)
                f_t = results.params
                f_t.name = date
                factor_ret_list.append(f_t)
                
                # B. Store Residuals (Specific Returns)
                # results.resid index is 'permno'. We need to re-attach the date.
                u_t = results.resid.to_frame('specific_ret')
                u_t['date'] = date 
                u_t.set_index('date', append=True, inplace=True) # -> (permno, date)
                resid_list.append(u_t)
                
            except Exception:
                continue

        # 3. Validation
        if not factor_ret_list:
             raise ValueError("Regressions failed for ALL dates. Check input data quality.")

        # 4. Concatenate Results
        self.factor_returns = pd.DataFrame(factor_ret_list)
        self.factor_returns.index.name = 'date'
        
        self.specific_returns = pd.concat(resid_list)
        # Swap levels to standard (permno, date) format if needed, though (permno, date) is fine for storage
        if self.specific_returns.index.names == ['permno', 'date']:
             self.specific_returns = self.specific_returns.swaplevel('permno', 'date')
        
        self.specific_returns.sort_index(inplace=True)
        
        print(f"Fama-MacBeth Complete. Estimated F & u for {len(self.factor_returns)} periods.")
    
    
    def predict_risk_matrices(self):
        """
        Forecasts the Risk Matrix components (F and Delta) using Exponentially Weighted Moving Average (EWMA).
        
        We assume that risk is persistent. Recent volatility predicts future volatility.
        """
        print("Forecasting Risk Matrices...")
        
        print("1. Forecast Factor Covariance Matrix (F)...")
        # 1. Forecast Factor Covariance Matrix (F)
        # We calculate the rolling EWMA covariance of the factor return history.
        # Result is a Panel: (Date, Factor, Factor)
        self.factor_cov_matrices = self.factor_returns.ewm(
            halflife=self.half_life, min_periods=self.half_life).cov()

        print("2. Forecast Idiosyncratic Variance (Delta)...")
        # 2. Forecast Idiosyncratic Variance (Delta)
        # We square the residuals to get specific variance
        # Unstack to (Date, Permno) to leverage pandas' time-series ewm() function
        u_wide = self.specific_returns['specific_ret'].unstack(level='permno')
        u_sq = u_wide ** 2
        
        # Apply EWMA smoothing
        self.idio_var = u_sq.ewm(halflife=self.half_life, 
                                 min_periods=self.half_life).mean()
        self.idio_vol = np.sqrt(self.idio_var)
        
        # Restack to (date, permno) long format for efficient storage
        self.idio_vol = self.idio_vol.stack(future_stack=True).to_frame('resid_vol').dropna()
        self.idio_vol.index.names = ['date', 'permno']
        self.idio_vol.sort_index(inplace=True)

        print("Risk forecasts complete.")

    def save_outputs(self):
        """
        Saves the forecasted risk components to Parquet.
        """
        print("Saving Risk Outputs...")
        
        # 1. Factor Covariance (Systematic Risk)
        path_cov = os.path.join(self.risk_output_folder_path, 'factor_cov_matrices.parquet')
        self.factor_cov_matrices.to_parquet(path_cov)
        
        # 2. Idiosyncratic Volatility (Specific Risk)
        path_vol = os.path.join(self.risk_output_folder_path, 'idio_vol.parquet')
        self.idio_vol.to_parquet(path_vol)
        
        print(f"Factor covariance matrix saved to: {path_cov}")
        print(f"Idiosyncratic volaility saved to: {path_vol}")
    
    def run_risk_pipeline(self):
        self.load_data()
        self.run_fama_macbeth()
        self.predict_risk_matrices()
        self.save_outputs()
    

if __name__ == "__main__":
    pass