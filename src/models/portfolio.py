"""
Module: Portfolio Optimization Engine

OBJECTIVE:
    Constructs optimal active portfolio weights (h_PA) that maximize Expected Active 
    Return for a given Active Risk budget (Tracking Error). The engine also generates 
    a comprehensive ex-ante risk decomposition report for the resulting portfolio.

METHODOLOGY:
    1. Alpha Calibration (Grinold-Kahn Refined Forecast):
       Converts standardized alpha signals (Z-scores) into expected active returns using 
       the Fundamental Law of Active Management:
       E[R] = IC * Specific_Volatility * Z-Score
       (This inversely weights assets by their idiosyncratic risk).

    2. Mean-Variance Optimization (MVO):
       Solves the Grinold-Kahn objective: Maximize (h^T * E[R]) - lambda * (h^T * V * h).
       The engine supports a dual-path solver:
       - Unconstrained Analytical: A blazingly fast linear algebra solver (V^-1 * E[R]) 
         that permits active cash drift.
       - Constrained Quadratic Programming (CVXPY): A rigorous numerical solver that 
         strictly enforces a dollar-neutral active portfolio (Sum of Active Weights = 0).

    3. Ex-Post Constraint Scaling:
       Applies fast analytical transformations to the optimal weights to enforce 
       real-world portfolio mandates:
       - Volatility Targeting: Scales weights so predicted risk perfectly matches 
         the Target Active Risk (e.g., 10% Tracking Error).
       - Gross Leverage Limits: Proportionally scales down exposures if the absolute 
         sum of active bets exceeds the maximum leverage threshold.

    4. Transaction Cost Control (Holdings Smoothing):
       Implements an autoregressive smoothing filter on the optimal weights. By blending 
       today's optimal target with yesterday's holdings, the engine severely limits 
       turnover (churn) while maintaining signal fidelity.

    5. Ex-Ante Risk Attribution:
       Decomposes the final portfolio's predicted risk into Systematic (Factor) vs. 
       Specific (Idiosyncratic) components, and audits the Active Beta to ensure the 
       constraints did not violate market neutrality.
"""
import os 
import numpy as np 
import pandas as pd 
import cvxpy as cp 
import hashlib
import json
from datetime import datetime

class PortfolioOptimizer:
    def __init__(self, 
                 data_paths: dict, 
                 portfolio_output_path: str, 
                 alpha_col: str, 
                 risk_aversion = 2.5, 
                 target_active_risk = 0.1, 
                 max_leverage = 3.0,
                 ic_estimate = 0.05,
                 turnover_smoothing = 0.0, 
                 max_turnover = None,
                 allow_active_cash = True): 
        
        self.data_paths = data_paths 
        self.portfolio_output_path = portfolio_output_path
        self.alpha_col = alpha_col
        self.risk_aversion = risk_aversion 
        self.target_active_risk = target_active_risk
        self.max_leverage = max_leverage
        self.ic_estimate = ic_estimate 
        self.turnover_smoothing = turnover_smoothing
        self.allow_active_cash = allow_active_cash 
        self.max_turnover = max_turnover
        """
        data_paths should include paths to 
        ={
            'alpha': 'alpha_signals.parquet',
            'exposures': 'X.parquet',
            'idio_vol': 'idio_vol.parquet',
            'factor_cov': 'factor_cov_matrices.parquet',
            'panel': 'panel_data.parquet'
        }
        """
    def load_inputs(self):
        
        data = {}

        for key, path in self.data_paths.items():
            if not os.path.exists(path): 
                raise FileNotFoundError(f"Missing {path}")
            df = pd.read_parquet(path)

            if key == 'factor_cov':
                data[key] = df
            else:
                if df.index.names != ['permno', 'date']:
                    df.reset_index(inplace = True)
                    df.set_index(['permno', 'date'], inplace = True)
                    df.sort_index(inplace = True)
                data[key] = df
        self.data = data
        return data
        
    def get_covariance_matrix(self, X, F, D_sq):
        """
        at each period t, 
        Reconstructs the full N x N covariance matrix (V) from the Factor Model.
        
        V = X * F * X.T + D
        Where:
        - X: Factor Exposures (N x K)
        - F: Factor Covariance (K x K)
        - D: Idiosyncratic Variance (Diagonal N x N)
        """

        # First, we make sure we are doing the matrix multiplication on shared factors 
        # Align dimensions
        common_factors = X.columns.intersection(F.index)
        X_sub = X[common_factors]
        F_sub = F.loc[common_factors, common_factors]
        
        # Convert to numpy for speed
        X_mat = X_sub.values.astype(float)
        F_mat = F_sub.values.astype(float)

        # Systematic component 
        sys_cov = X_mat @ F_mat @ X_mat.T # first term RHS of (3A.2) in G&K
        # Specific cariance component
        spec_cov = np.diag(D_sq.values.flatten().astype(float)) # this is delta in eq (3A.2) in G&K

        V_raw =sys_cov + spec_cov 
        # to avoid errors in optimization using cvxpy, 
        # we explicitly enforce symmetry to fix floating-point rounding errors
        V_symmetric = (V_raw + V_raw.T) / 2.0
        return V_symmetric, spec_cov, X_sub, F_sub


    def optimize_single_period(self, date, data, h_PA_prev = None):
        """
        Performs Mean-Variance Optimization for a single date.
        
        Returns:
            h_PA: Optimal active weights (sum to 0).
        """
        # 1. Retrieve Data Slices

        try:
            alpha_series = data['alpha'].xs(date, level = 'date')[self.alpha_col].dropna()
            exposures = data['exposures'].xs(date, level = 'date').dropna()
            F = data['factor_cov'].xs(date, level = 'date').dropna()
            idio_vol = data['idio_vol'].xs(date, level = 'date').dropna()
            mkt_cap = data['panel'].xs(date, level = 'date')['mkt_cap'].dropna()
        except KeyError: 
            return None 

        # Intersection of valid assets across all inputs
        valid_assets = (alpha_series.index
                                .intersection(exposures.index)
                                .intersection(idio_vol.index)
                                .intersection(mkt_cap.index))
        
            
        if len(valid_assets) < 10: return None # Skip if universe is too small

        if h_PA_prev is not None:
            h_PA_prev_aligned = h_PA_prev.reindex(valid_assets).fillna(0.0).values
        else:
            h_PA_prev_aligned = np.zeros(len(valid_assets))
        

        # Filter to valid universe
        alpha_series = alpha_series.loc[valid_assets]
        X = exposures.loc[valid_assets]
        D_sq = idio_vol.loc[valid_assets] ** 2 
        mc = mkt_cap.loc[valid_assets]

        # 2. Construct Covariance (V)
        V, delta, X_aligned, F_aligned = self.get_covariance_matrix(X, F, D_sq)
             
        
        # 3. Scale Alpha
        # We convert Z-Score Alpha into Expected Exceptional Returns.
        # Forecast = IC * Volatility * Z-Score
        # alpha_scaled = alpha_series * avg_vol * self.ic_estimate
        # we are going to use G&K formula from chapter 10
        # as a result, the optimal weight of a stock becomes inversely proportional to its volatility.
        alpha_scaled = alpha_series * idio_vol.loc[valid_assets]['resid_vol'] * self.ic_estimate


        # --- DEBUGGING PRINTS: Inputs ---
        # 1. Check for missing values in our inputs
        alpha_nans = alpha_scaled.isna().sum()
        v_nans = np.isnan(V.astype(float)).sum()
        
        # 2. Check matrix dimensions
        print(f"\n--- Debugging Date: {date.date()} ---")
        print(f"Valid Assets Length: {len(valid_assets)}")
        print(f"Alpha NaNs: {alpha_nans} | V Matrix NaNs: {v_nans}")
        print(f"V Shape: {V.shape} | Alpha Shape: {alpha_scaled.shape}")

        scaler = 1 / (2 * self.risk_aversion)
        
        # 4. Solve Optimization
        N = len(valid_assets)
        is_unconstrained = (self.allow_active_cash) and (self.max_turnover is None)

        if is_unconstrained:

            # Equation: h = (1/2*lambda) * V_inv * alpha
            try: 
                h_PA_raw = scaler * np.linalg.solve(V.astype(float), alpha_scaled.values.astype(float))
            except np.linalg.LinAlgError: return None 

            # --- DEBUGGING PRINTS: Solver Output ---
            h_raw_nans = np.isnan(h_PA_raw).sum()
            print(f"h_PA_raw NaNs (Output from solver): {h_raw_nans}")
            
            # --- DEBUGGING PRINTS: Constraint Math ---
            std_PA_raw = np.sqrt(h_PA_raw.T @ V @ h_PA_raw)
            print(f"Raw Portfolio Volatility (std_PA_raw): {std_PA_raw}")

        else: 
            # The constrained, numerical Portfolio Y (Slower, but enforces e_PA = 0)
            # Define the optimization variable (N x 1 vector)
            h = cp.Variable(N)

            # Define the objective function: Maximize (alpha^T * h) - lambda * (h^T * V * h)
            # cvxpy requires formulating as a Minimization problem
            expected_return = alpha_scaled.values.T @ h
            risk_penalty = self.risk_aversion * cp.quad_form(h, cp.psd_wrap(V.astype(float)))
            
            objective = cp.Maximize(expected_return - risk_penalty)

            constraints= []

            # if no active cash is allowed, append the "No Active Cash" constraint: Sum of active weights = 0
            if not self.allow_active_cash:
                constraints.append(cp.sum(h) == 0)
            # if a max_turnover limit is set, append the condition to the constraints 
            if self.max_turnover is not None:
                max_weight_change = self.max_turnover * 2.0
                # Take my new target weights (h), subtract my old weights (h_prev), 
                # take the absolute value of every single trade, 
                # sum them all up (the L1 Norm), 
                # and mathematically force that total to be less than my turnover limit
                constraints.append(cp.norm1(h - h_PA_prev_aligned) <= max_weight_change)

            # Define and solve the problem
            prob = cp.Problem(objective, constraints)
            
            try:
                # we are going to use ECOS or OSQP solvers 
                prob.solve(solver=cp.OSQP) 
                
                if h.value is None: # Solver failed to find a solution
                    return None
                    
                h_PA_raw = h.value
                
            except cp.error.SolverError:
                return None

        # ------------- Appling Constraints (Scaling)  ----------------
        # CONSTRAINT 1: Volatility Targeting
        
        # Volatility Targeting
        # We scale the portfolio so its predicted risk equals the target risk.
        std_PA_raw = np.sqrt(h_PA_raw.T @ V @ h_PA_raw)
        target_monthly = self.target_active_risk / np.sqrt(12)

        vol_scalar = target_monthly / std_PA_raw if std_PA_raw > 1e-6 else 0.0
        h_PA_scaled = h_PA_raw * vol_scalar

        # CONSTRAINT 2: Maximum Gross Leverage
        # Leverage Constraint
        # If gross leverage exceeds the limit, scale down everything proportionally.
        # Gross Leverage is the sum of absolute active weights (Longs + Shorts)
        # if gross leverage exceeds the limit, scale down the active positions proportionally 

        current_leverage = np.abs(h_PA_scaled).sum()
        if current_leverage > self.max_leverage:
            lev_scalar = self.max_leverage / current_leverage
            h_PA_scaled = h_PA_scaled * lev_scalar
    
        h_PA = pd.Series(h_PA_scaled, index=valid_assets) 
        return h_PA, valid_assets, V, delta,  X_aligned, F_aligned, mc 

    def run_optimization(self):
        """
        Runs the optimizer sequentially over the entire backtest history.
        Manages state (previous holdings) for turnover smoothing.
        """
        print("Loading inputs...")
        data = self.load_inputs()
        dates = data['alpha'].index.get_level_values('date').unique().sort_values()
        
        all_holdings = [] # A list of dataframes each for holdings on a specific date 
        attribution_stats = [] # A list of dictionaries, each for a set of attributes of the portfolio on a specific date 

        h_PA_prev = None # We track previous waeight for smooothing
        print("Running the portfolio optimization for each date")
        for date in dates:
            res = self.optimize_single_period(date, data)
            if res is not None:
                h_PA_target, assets, V, delta, X, F, mc = res

                # ------- holdings smoothing --------
                # This acts as a dampener on trading.
                # h_final = (1 - smoothing_factor) * h_optimal + (smoothing_factor) * h_previous
                
                if h_PA_prev is not None and self.turnover_smoothing > 0:
                    # first, it's possible that the h_PA_prev doesnt have position in certain stocks that h_PA_target does. 
                    # so we reindex using the asset index which the optimize_single_preiod returns 
                    h_PA_prev_aligned = h_PA_prev.reindex(assets).fillna(0)
                    
                    # smoothing the h_PA_target
                    h_PA = (1 - self.turnover_smoothing) * h_PA_target + \
                               (self.turnover_smoothing) * h_PA_prev_aligned
                else: 
                    h_PA = h_PA_target
                
                # now let's update the previous active holdings with current holdings 
                # so that they are ready for the next iteration 
                h_PA_prev = h_PA.copy()

                # (3A.12)
                var_PA = h_PA.T @ V @ h_PA

                # (3A.11)
                x_PA = X.values.T @ h_PA 

                # (3A.9)
                var_PA_factor = x_PA .T @ F.values @ x_PA
                var_PA_spec = var_PA - var_PA_factor

                if var_PA > 1e-9:
                    spec_ratio = var_PA_spec / var_PA
                    factor_ratio = var_PA_factor / var_PA
                else:
                    spec_ratio, factor_ratio = 0.0, 0.0
                
                h_B = mc / mc.sum()
                h_P = h_B + h_PA
                
                # the extent of investment 
                e_P = h_P.sum()
                e_PA = h_PA.sum()
                
                # calcualting benchmark Variance
                var_B  = h_B.T @ V @ h_B

                #(3A.9)
                var_P  = h_P.T @ V @ h_P
                x_P = X.values.T @ h_P

                # calcualting the beta vector using eq (3A.13) in G&K 
                beta = (1 / var_B) * V @ h_B 
                # (3A.17)
                beta_P = h_P.T @ beta 
                beta_PA = h_PA.T @ beta 

                # (3A.14):
                x_B = X.values.T @ h_B

                if var_B > 1e-6: 
                    b = (1/var_B) * F.values @ x_B
                    # (3A.15)
                    d = (1/var_B) * delta @ h_B
                    # (3A.17)
                    beta_P_factor = x_P.T @ b 
                    beta_P_specific =  h_P.T @ d

                # (3A.18)
                var_systematic = beta_P**2 * var_B
                var_residual = var_P - var_systematic 
                
                # (3A.19)
                #V_sys = var_B * beta @ beta.T 
                #V_residual = V - V_sys

                df_res = pd.DataFrame(data = {'h_active': h_PA, 
                                              'h_benchmark': h_B,
                                              'h_portfolio': h_P}, 
                                              index = assets)
                df_res['date'] = date
                df_res.set_index('date', append = True, inplace = True)    
                df_res = df_res.reorder_levels(['permno','date'])
                all_holdings.append(df_res)

                attribution_stats.append({
                    'date': date,
                    # --- METADATA (Parameters) ---
                    'alpha_signal': self.alpha_col,
                    'target_active_risk': self.target_active_risk,
                    'max_leverage': self.max_leverage,
                    'turnover_smoothing': self.turnover_smoothing,
                    'allow_active_cash': self.allow_active_cash,
                    # --- EXPOSURE METRICS ---
                    'net_active_exposure': e_PA,
                    'gross_market_exposure': e_P,

                    # --- RISK METRICS ---
                    'predicted_active_risk': np.sqrt(var_PA),
                    'specific_risk_pct': spec_ratio,
                    'factor_risk_pct': factor_ratio, 
                    'portfolio_beta': beta_P, 
                    'active_beta': beta_PA, 
                    'predicted_portfolio_volatility': np.sqrt(var_P),
                    'predicted_benchmark_volatility': np.sqrt(var_B), 
                    'portfolio_systematic_risk_pct': var_systematic / var_P,
                    'portfolio_residual_risk_pct': var_residual / var_P})
        
        print("Consolidating the optimization results from all periods")
        # Consolidate and Save
        final_holdings = pd.concat(all_holdings)
        final_stats = pd.DataFrame(attribution_stats).set_index('date')

        print("Generating Run ID and saving outputs...")
        
        # --- 1. Define Parameters Dictionary ---
        params = {
            'alpha_col': self.alpha_col,
            'target_active_risk': self.target_active_risk,
            'allow_active_cash': self.allow_active_cash,
            'max_leverage': self.max_leverage,
            'turnover_smoothing': self.turnover_smoothing,
            'max_turnover': self.max_turnover
        }

        # --- 2. Generate Hash and Run ID ---
        param_string = json.dumps(params, sort_keys=True)
        short_hash = hashlib.md5(param_string.encode()).hexdigest()[:7]
        run_id = f"run_{datetime.now().strftime('%Y%m%d')}_{short_hash}"

        # --- 3. Construct Clean Paths ---
        holdings_path = os.path.join(self.portfolio_output_path, f"holdings_{run_id}.parquet")
        stats_path = os.path.join(self.portfolio_output_path, f"stats_{run_id}.parquet")
        config_path = os.path.join(self.portfolio_output_path, f"config_{run_id}.json")

        # --- 4. Save to Disk ---
        final_holdings.to_parquet(holdings_path)
        final_stats.to_parquet(stats_path)
        
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"Optimization Complete. Outputs saved under Run ID: {run_id}")
        
        # Return the run_id so the backtest module can catch it!
        return run_id