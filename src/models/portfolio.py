"""
Module: Portfolio Optimization Engine

OBJECTIVE:
    Constructs the optimal active portfolio weights ($h^*$) that maximize 
    Expected Active Return for a given level of Active Risk.

METHODOLOGY:
    1. Mean-Variance Optimization:
       We solve the unconstrained Grinold-Kahn equation:
       h_raw = (1 / 2*lambda) * V^(-1) * alpha
       
       Where:
       - V     = Asset Covariance Matrix (Factor + Idiosyncratic)
       - alpha = Expected Active Returns (Scaled Forecasts)
       - lambda = Risk Aversion parameter

    2. Constraint Application (Ex-Post Scaling):
       Instead of a slow quadratic programming solver (QP), we use a fast 
       analytical approach:
       - Step A: Solve unconstrained.
       - Step B: Scale weights to hit Target Active Risk (e.g., 5% tracking error).
       - Step C: Scale weights (if necessary) to respect Gross Leverage limits.

    3. Transaction Cost Control (Holdings Smoothing):
       We implement an autoregressive smoothing filter on the optimal weights 
       to prevent excessive turnover (churn) while maintaining signal fidelity.
"""

import os
import pandas as pd
import numpy as np

class PortfolioOptimizer:
    """
    The engine that combines Alpha and Risk to construct optimal portfolios.
    
    Attributes:
        target_active_risk (float): The desired Annualized Tracking Error (e.g., 0.05 = 5%).
        max_leverage (float): Gross Leverage Limit (Long + Short). 3.0 = 300% exposure.
        turnover_smoothing (float): Autoregressive decay factor for trading.
                                    0.0 = Trade fully to target (High Turnover).
                                    0.5 = Trade 50% of the distance (Low Turnover).
    """
    
    def __init__(self, data_dir, 
                 risk_aversion=0.5, 
                 target_active_risk=0.10, 
                 max_leverage=3.0, 
                 ic_estimate=0.05,
                 alpha_col='alpha_Composite_final',
                 turnover_smoothing=0.0): 
        """
        Args:
            turnover_smoothing (float): Fraction of the previous position to retain. 
                                        Acts as a proxy for transaction cost penalties.
        """
        self.data_dir = data_dir
        self.risk_aversion = risk_aversion
        self.target_active_risk = target_active_risk
        self.max_leverage = max_leverage
        self.ic_estimate = ic_estimate
        self.alpha_col = alpha_col
        self.turnover_smoothing = turnover_smoothing

    def load_inputs(self):
        """Loads all required components: Alpha, Exposures, Risk Model."""
        print(f"Loading Optimization Inputs (Signal: {self.alpha_col})...")
        paths = {
            'alpha': os.path.join(self.data_dir, 'alpha_signals.parquet'),
            'exposures': os.path.join(self.data_dir, 'factor_exposures.parquet'),
            'idio_vol': os.path.join(self.data_dir, 'idio_vol.parquet'),
            'factor_cov': os.path.join(self.data_dir, 'factor_cov_matrices.parquet'),
            'panel': os.path.join(self.data_dir, 'panel_data.parquet')
        }
        data = {}
        for key, path in paths.items():
            if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")
            df = pd.read_parquet(path)
            
            # Special handling for Factor Covariance (it's not a panel)
            if key == 'factor_cov': 
                data[key] = df
            else:
                # Ensure Panel Alignment
                if df.index.names != ['permno', 'date']:
                    df.reset_index(inplace=True)
                    df.set_index(['permno', 'date'], inplace=True)
                    df.sort_index(inplace=True)
                data[key] = df
        return data

    def get_covariance_matrix(self, X, F, D_sq):
        """
        Reconstructs the full N x N covariance matrix (V) from the Factor Model.
        
        V = X * F * X.T + D
        Where:
        - X: Factor Exposures (N x K)
        - F: Factor Covariance (K x K)
        - D: Idiosyncratic Variance (Diagonal N x N)
        """
        # Align Dimensions
        common_factors = X.columns.intersection(F.index)
        X_sub, F_sub = X[common_factors], F.loc[common_factors, common_factors]
        
        # Convert to numpy for speed
        X_mat, F_mat = X_sub.values.astype(float), F_sub.values.astype(float)
        
        # Systematic Component
        sys_cov = X_mat @ F_mat @ X_mat.T
        
        # Idiosyncratic Component
        spec_cov = np.diag(D_sq.values.astype(float))
        
        return sys_cov + spec_cov, X_sub, F_sub 

    def optimize_single_period(self, date, data):
        """
        Performs Mean-Variance Optimization for a single date.
        
        Returns:
            h_PA: Optimal active weights (sum to 0).
        """
        # 1. Retrieve Data Slices
        try:
            alpha_series = data['alpha'].xs(date, level='date')[self.alpha_col]
            exposures = data['exposures'].xs(date, level='date')
            idio_vol = data['idio_vol'].xs(date, level='date')['resid_vol']
            mkt_cap = data['panel'].xs(date, level='date')['mkt_cap']
            F = data['factor_cov'].loc[date]
        except KeyError: return None

        # Intersection of valid assets across all inputs
        valid_assets = alpha_series.index.intersection(exposures.index).intersection(idio_vol.index).intersection(mkt_cap.index)
        if len(valid_assets) < 10: return None # Skip if universe is too small

        # Filter to valid universe
        alpha = alpha_series.loc[valid_assets]
        X = exposures.loc[valid_assets]
        D_sq = idio_vol.loc[valid_assets] ** 2
        mc = mkt_cap.loc[valid_assets]
        
        # 2. Construct Covariance (V)
        V, X_aligned, F_aligned = self.get_covariance_matrix(X, F, D_sq)
        
        # 3. Scale Alpha
        # We convert Z-Score Alpha into Expected Returns.
        # Forecast = IC * Volatility * Z-Score
        avg_vol = idio_vol.loc[valid_assets].mean()
        alpha_scaled = alpha * avg_vol * self.ic_estimate
        
        # 4. Solve Unconstrained Optimization
        # Equation: h = (1/2*lambda) * V_inv * alpha
        scalar = 1.0 / (2 * self.risk_aversion)
        
        try:
            # use solve() instead of inv() for numerical stability
            h_PA_raw = scalar * np.linalg.solve(V.astype(np.float64), alpha_scaled.values.astype(np.float64))
        except np.linalg.LinAlgError: return None

        # 5. Apply Constraints (Scaling)
        
        # A. Volatility Targeting
        # We scale the portfolio so its predicted risk equals the target risk.
        raw_risk = np.sqrt(h_PA_raw.T @ V @ h_PA_raw)
        target_monthly = self.target_active_risk / np.sqrt(12)
        
        vol_scale = target_monthly / raw_risk if raw_risk > 1e-6 else 0.0
        current_h = h_PA_raw * vol_scale

        # B. Leverage Constraint
        # If gross leverage exceeds the limit, scale down everything proportionally.
        current_lev = np.sum(np.abs(current_h))
        lev_scale = self.max_leverage / current_lev if current_lev > self.max_leverage else 1.0
            
        h_PA = current_h * lev_scale
        
        return h_PA, valid_assets, V, X_aligned, F_aligned, mc

    def run_optimization(self):
        """
        Runs the optimizer sequentially over the entire backtest history.
        Manages state (previous holdings) for turnover smoothing.
        """
        data = self.load_inputs()
        dates = data['alpha'].index.get_level_values('date').unique().sort_values()
        
        all_holdings = []
        attribution_stats = []
        prev_h_active = None # Track previous weights for smoothing
        
        print(f"Optimizing {self.alpha_col} (Smoothing={self.turnover_smoothing:.1f})...")
        
        for date in dates:
            res = self.optimize_single_period(date, data)
            if res is not None:
                h_target, assets, V, X, F, mc = res
                
                # --- HOLDINGS SMOOTHING ---
                # This acts as a dampener on trading.
                # h_final = (1 - smooth) * h_optimal + (smooth) * h_previous
                if prev_h_active is not None and self.turnover_smoothing > 0:
                    # Align previous holdings to current universe
                    h_prev_aligned = prev_h_active.reindex(assets).fillna(0.0)
                    
                    h_final = (1.0 - self.turnover_smoothing) * h_target + \
                              (self.turnover_smoothing) * h_prev_aligned.values
                else:
                    h_final = h_target
                
                # Update State for next loop
                prev_h_active = pd.Series(h_final, index=assets)
                # -------------------------------
                
                # Risk Decomposition Stats
                total_var = h_final.T @ V @ h_final
                active_exp = X.values.T @ h_final 
                factor_var = active_exp.T @ F.values @ active_exp
                spec_var = total_var - factor_var
                
                # Avoid div by zero
                if total_var > 1e-9:
                    spec_ratio = spec_var / total_var
                    fact_ratio = factor_var / total_var
                else:
                    spec_ratio, fact_ratio = 0.0, 0.0

                # Construct Portfolio Components
                h_B = mc / mc.sum() # Benchmark Weights (Cap Weighted)
                h_P = h_B + h_final # Total Portfolio Weights

                # Store Holdings
                df_res = pd.DataFrame({
                    'h_active': h_final,
                    'h_benchmark': h_B,
                    'h_portfolio': h_P,
                }, index=assets)
                
                df_res['date'] = date
                df_res.set_index('date', append=True, inplace=True)
                df_res = df_res.swaplevel('permno', 'date')
                all_holdings.append(df_res)
                
                # Store Attribution
                attribution_stats.append({
                    'date': date,
                    'predicted_active_risk': np.sqrt(total_var),
                    'specific_risk_pct': spec_ratio,
                    'factor_risk_pct': fact_ratio
                })
                
        # Consolidate and Save
        final_holdings = pd.concat(all_holdings)
        final_stats = pd.DataFrame(attribution_stats).set_index('date')
        
        final_holdings.to_parquet(os.path.join(self.data_dir, 'optimal_holdings.parquet'))
        final_stats.to_csv(os.path.join(self.data_dir, 'risk_attribution.csv'))
        print("Optimization Complete.")