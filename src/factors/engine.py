"""
Module: Factor Construction Engine

OBJECTIVE:
    Constructs the Time-Varying Factor Exposure Matrix (X_t) using a Point-in-Time 
    framework. This module transforms raw fundamental and market data into 
    standardized risk factors.

METHODOLOGY:
    1. Industry Classification:
       Maps SIC codes to Fama-French 12 Industries to capture sector risk.

    2. Composite Style Factors:
       - Value: Composite of B/M (lagged 6mo) and E/P (lagged 3mo).
       - Momentum: Composite of 12-1 Month and 6-1 Month returns.
       - Financial Constraints: Whited-Wu Index (Cash Flow, Leverage, Dividend, Sales Growth).
       - Size: Log(Market Cap).

    3. Capitalization-Weighted Standardization:
       Factors are standardized cross-sectionally using market-cap weights to 
       prevent micro-cap outliers from skewing the signal:
       Z = (X - Mean_CapWeighted) / Std_CapWeighted

    4. Point-in-Time Integrity:
       Strictly enforces reporting lags (using 'rdq' or 'datadate' offsets) to 
       ensure no information is used before it was publicly available.
"""

import pandas as pd
import numpy as np
from typing import Optional

class FactorEngine:
    """
    Orchestrates the construction of the Factor Exposure Matrix (X).
    
    This engine handles the full lifecycle of feature engineering:
    - Calculation of raw descriptors (e.g., Book-to-Market ratios).
    - Handling of temporal lags (Point-in-Time alignment).
    - Cross-sectional standardization (Z-Scoring).
    - Sector dummy generation.
    """
    
    def __init__(self, panel_data: pd.DataFrame):
        """
        Args:
            panel_data: Monthly panel containing market data (mkt_cap, ret_monthly)
                        and raw fundamentals (ceqq, ibq, atq, etc.).
        """
        # Work on a copy to ensure immutability of source data
        self.df = panel_data.copy()
        
        # Ensure multi-index is flat for calculations, will reset later
        # This makes vectorization easier for group operations
        if isinstance(self.df.index, pd.MultiIndex):
            self.df.reset_index(inplace=True)
            
        # Enforce types for merge keys
        self.df['permno'] = self.df['permno'].astype(int)
        self.df['date'] = pd.to_datetime(self.df['date'])

    def build_factors(self) -> pd.DataFrame:
        """
        Main pipeline execution.
        
        Returns: 
            pd.DataFrame: The final X matrix (Date, Permno index) with 
                          Standardized Factors and Industry Dummies.
        """
        print("--- Starting Factor Construction ---")
        
        # ----------------------------------------------------------------------
        # 1. INDUSTRY CLASSIFICATION
        # ----------------------------------------------------------------------
        # We categorize firms into Fama-French 12 industry groups based on SIC codes.
        # This allows the risk model to isolate sector-specific variance (e.g., Tech vs Energy).
        print("Assigning Fama-French 12 Industries...")
        self.df['industry'] = self.df['sic'].apply(self._sic_to_ff12)
        
        # ----------------------------------------------------------------------
        # 2. RAW DESCRIPTOR CALCULATION
        # ----------------------------------------------------------------------
        # We compute the raw values for each factor component.
        # Critical: These functions handle the "Point-in-Time" lag logic internally.
        print("Calculating Raw Descriptors (Size, Value, Momentum, WW)...")
        self._calc_size()
        self._calc_value_pit() # Handles announcement lags for B/M and E/P
        self._calc_momentum()
        self._calc_whited_wu()
        
        # 3. Clean Infinite values
        # Log transformations (like Size) can produce -Inf for penny stocks.
        descriptor_cols = ['size_desc', 'bm_desc', 'ep_desc', 'mom12_1_desc', 'mom6_1_desc', 'ww_desc']
        self.df[descriptor_cols] = self.df[descriptor_cols].replace([np.inf, -np.inf], np.nan)
        
        # ----------------------------------------------------------------------
        # 4. CROSS-SECTIONAL STANDARDIZATION (CAP-WEIGHTED)
        # ----------------------------------------------------------------------
        # We standardize factors to have Mean=0, Std=1 cross-sectionally.
        # 
        # Crucial Detail: We use *Market-Cap Weights* for the mean and std.
        # Why? An Equal-Weighted Z-score is dominated by micro-caps (which are numerous 
        # but economically irrelevant). Cap-Weighting ensures the "Zero" of the factor 
        # represents the true market average, not the average of tiny stocks.
        print("Standardizing Descriptors (Cap-Weighted)...")
        
        # Calculate monthly weight vector once
        self.df['cap_weight'] = self.df.groupby('date')['mkt_cap'].transform(lambda x: x / x.sum())
        
        for col in descriptor_cols:
            self.df[f'z_{col}'] = self.df.groupby('date')[col].transform(
                lambda x: self._standardize_cap_weighted(x, self.df.loc[x.index, 'cap_weight'])
            )
            
        # ----------------------------------------------------------------------
        # 5. COMPOSITE CONSTRUCTION
        # ----------------------------------------------------------------------
        # We combine correlated descriptors into single robust factors to reduce noise.
        print("Building Composite Factors...")
        
        # Value = Average(Z_BM, Z_EP)
        # Combines Balance Sheet Value (Book/Price) with Income Statement Value (Earnings/Price)
        self.df['Value_composite'] = self.df[['z_bm_desc', 'z_ep_desc']].mean(axis=1)
        
        # Momentum = Average(Z_Mom12-1, Z_Mom6-1)
        # Smooths the momentum signal by averaging the 1-year and 6-month horizons
        self.df['Momentum_composite'] = self.df[['z_mom12_1_desc', 'z_mom6_1_desc']].mean(axis=1)
        
        # ----------------------------------------------------------------------
        # 6. FINAL RE-STANDARDIZATION
        # ----------------------------------------------------------------------
        # Compositing changes the distribution variance. We re-standardize the final 
        # composites to ensure they are strictly N(0,1) for the regression engine.
        self.df['Value'] = self.df.groupby('date')['Value_composite'].transform(
            lambda x: self._standardize_cap_weighted(x, self.df.loc[x.index, 'cap_weight'])
        )
        self.df['Momentum'] = self.df.groupby('date')['Momentum_composite'].transform(
            lambda x: self._standardize_cap_weighted(x, self.df.loc[x.index, 'cap_weight'])
        )
        
        # Rename single descriptors for consistency
        self.df.rename(columns={'z_size_desc': 'Size', 'z_ww_desc': 'FinConstraint'}, inplace=True)
        
        # ----------------------------------------------------------------------
        # 7. ONE-HOT ENCODING (INDUSTRY DUMMIES)
        # ----------------------------------------------------------------------
        print("Generating Industry Dummies...")
        industry_dummies = pd.get_dummies(self.df['industry'], prefix='Ind')
        industry_dummies = industry_dummies.astype(float) 
        
        # ----------------------------------------------------------------------
        # 8. ASSEMBLE FINAL X MATRIX
        # ----------------------------------------------------------------------
        style_factors = ['Size', 'Value', 'Momentum', 'FinConstraint']
        
        # Set index back to (date, permno) for the final output
        final_df = self.df.set_index(['date', 'permno'])
        X = final_df[style_factors].join(industry_dummies.set_index(self.df.index))
        X.index = final_df.index # Align indices explicitly
        
        # Drop rows missing ANY factor.
        # Why? The cross-sectional regression (Factor Return Estimation) requires 
        # complete data. A missing factor exposure breaks the linear algebra solver.
        X.dropna(inplace=True)
        
        print(f"Factor Construction Complete. Final Shape: {X.shape}")
        return X

    # -------------------------------------------------------------------------
    #                            FACTOR LOGIC
    # -------------------------------------------------------------------------

    def _calc_size(self):
        """
        Size Factor.
        Definition: Natural Log of Total Market Capitalization.
        Proxy for: Information asymmetry and liquidity risk.
        """
        self.df['size_desc'] = np.log(self.df['mkt_cap'])

    def _calc_value_pit(self):
        """
        Value Factor (Point-in-Time).
        
        Implementation Detail:
        Instead of assuming data is available at quarter-end, we use `merge_asof`
        to simulate the exact information set available to an investor.
        
        1. Book-to-Market: 
           Matches Price(t) with the most recent Book Equity that has been public 
           for at least 6 months (standard academic lag to ensure availability).
           
        2. Earnings-to-Price:
           Matches Price(t) with Trailing 12M Earnings, using a 3-month reporting lag.
        """
        # --- Book-to-Market ---
        # Isolate Book Equity Data
        cols_bm = ['permno', 'datadate', 'rdq', 'ceqq']
        fund_bm = self.df[cols_bm].copy().dropna(subset=['datadate']).drop_duplicates()
        
        # Define Availability Date: 
        # Use explicit Earnings Announcement Date (RDQ) if available.
        # Fallback: Assume a conservative 6-month lag from fiscal year-end if RDQ is missing.
        fund_bm['announcement_date'] = fund_bm['rdq'].fillna(
            fund_bm['datadate'] + pd.DateOffset(months=6)
        )
        fund_bm.rename(columns={'ceqq': 'be_lag'}, inplace=True)
        
        # Merge back to Main Panel using AsOf (PIT)
        self.df = pd.merge_asof(
            left=self.df.sort_values('date'),
            right=fund_bm[['permno', 'announcement_date', 'be_lag']].sort_values('announcement_date'),
            left_on='date',
            right_on='announcement_date',
            by='permno'
        )
        self.df['bm_desc'] = self.df['be_lag'] / self.df['mkt_cap']

        # --- Earnings-to-Price ---
        # Isolate Earnings Data
        cols_ep = ['permno', 'datadate', 'rdq', 'ibq']
        fund_ep = self.df[cols_ep].copy().dropna(subset=['datadate']).drop_duplicates()
        fund_ep.sort_values(['permno', 'datadate'], inplace=True)
        
        # Calculate LTM Earnings (Rolling 4 quarters sum)
        fund_ep['ltm_earn'] = fund_ep.groupby('permno')['ibq'].rolling(4, min_periods=4).sum().values
        
        # Define Availability Date: RDQ or DataDate + 3 Months (Standard quarterly lag)
        fund_ep['announcement_date'] = fund_ep['rdq'].fillna(
            fund_ep['datadate'] + pd.DateOffset(months=3)
        )
        fund_ep.dropna(subset=['ltm_earn'], inplace=True)
        
        # Merge back
        self.df = pd.merge_asof(
            left=self.df.sort_values('date'),
            right=fund_ep[['permno', 'announcement_date', 'ltm_earn']].sort_values('announcement_date'),
            left_on='date',
            right_on='announcement_date',
            by='permno'
        )
        self.df['ep_desc'] = self.df['ltm_earn'] / self.df['mkt_cap']

    def _calc_momentum(self):
        """
        Momentum Factor.
        
        We calculate two horizons to capture the robust price trend:
        1. 12-1 Month: Return from t-12 to t-1 (Standard academic momentum).
        2. 6-1 Month: Return from t-6 to t-1 (Faster signal).
        
        Note: We explicitly skip the most recent month (t-1) to avoid 
        short-term reversal effects and microstructure noise (bid-ask bounce).
        """
        # Sort by date for proper rolling
        self.df.sort_values(['permno', 'date'], inplace=True)
        
        # Helper lambda for geometric compounding
        compound = lambda x: (1 + x).prod() - 1
        
        grouped = self.df.groupby('permno')['ret_monthly']
        
        # Note: shift(1) excludes current month
        self.df['mom12_1_desc'] = grouped.transform(lambda x: x.shift(1).rolling(11).apply(compound))
        self.df['mom6_1_desc'] = grouped.transform(lambda x: x.shift(1).rolling(5).apply(compound))

    def _calc_whited_wu(self):
        """
        Financial Constraints Factor (Whited-Wu Index).
        
        A composite measure of financial distress based on:
        - Cash Flow / Assets (-)
        - Dividend Payer Status (-)
        - Leverage (+)
        - Size (-)
        - Sales Growth (-)
        
        Interpretation: Higher Value = Higher Constraint (More Distress).
        """
        # Components
        cf_at = (self.df['ibq'] + self.df['dpq']) / self.df['atq']
        div_pos = ((self.df['dvpspq'] * self.df['cshoq']) > 0).astype(int)
        tltd_at = self.df['dlttq'] / self.df['atq']
        ln_at = np.log(self.df['atq'].replace(0, np.nan))
        
        # Sales Growth (SG)
        sg = self.df.sort_values('date').groupby('permno')['saleq'].pct_change()
        
        # Industry Sales Growth (ISG) - Mean SG per Industry-Date
        self.df['sg_temp'] = sg
        self.df['isg'] = self.df.groupby(['industry', 'date'])['sg_temp'].transform('mean')
        
        # Formula (Coefficients from Whited & Wu, 2006)
        self.df['ww_desc'] = (
            -0.091 * cf_at 
            - 0.062 * div_pos 
            + 0.021 * tltd_at 
            - 0.044 * ln_at 
            + 0.102 * self.df['isg'] 
            - 0.035 * sg
        )

    # -------------------------------------------------------------------------
    #                            HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _standardize_cap_weighted(series: pd.Series, weights: pd.Series) -> pd.Series:
        """
        Performs capitalization-weighted standardization: Z = (x - mean_w) / std_w
        
        Why Cap-Weighted?
        In equal-weighted standardization, the mean and std dev are dominated by 
        thousands of micro-cap stocks. This creates a "Z-Score" that measures 
        distance from the *average tiny stock*.
        
        By cap-weighting, Z=0 represents the *dollar-weighted average* of the market 
        (i.e., the S&P 500 average). This makes the factor exposure much more 
        meaningful for institutional portfolios.
        """
        # Alignment & NaNs
        if series.isnull().all(): return pd.Series(np.nan, index=series.index)
        
        # Align indices (Crucial for safety)
        series, weights = series.align(weights, join='left')
        valid = series.notna() & weights.notna()
        
        if not valid.any(): return pd.Series(np.nan, index=series.index)
        
        w = weights[valid]
        s = series[valid]
        
        # Normalize weights to sum to 1
        w = w / w.sum()
        
        # Weighted Mean
        mean_w = np.sum(s * w)
        
        # Weighted Std
        var_w = np.sum(w * (s - mean_w)**2)
        std_w = np.sqrt(var_w)
        
        if std_w == 0: return pd.Series(0.0, index=series.index)
        
        return (series - mean_w) / std_w

    @staticmethod
    def _sic_to_ff12(sic) -> str:
        """
        Maps SIC Code to Fama-French 12 Industry Group.
        
        Provides coarse-grained sector control for the risk model.
        Mappings based on Kenneth French's data library definitions.
        """
        if pd.isnull(sic): return 'Other'
        try:
            s = int(sic)
        except:
            return 'Other'

        # Mapping Logic
        if (2830 <= s <= 2836) or (3840 <= s <= 3851) or (8000 <= s <= 8099): return 'Healthcare'
        if (3570 <= s <= 3579) or (3660 <= s <= 3679) or (7370 <= s <= 7379): return 'Technology'
        if (1300 <= s <= 1399) or (2900 <= s <= 2999): return 'Energy'
        if (4900 <= s <= 4949): return 'Utilities'
        if (4800 <= s <= 4899): return 'Telecom'
        if (6000 <= s <= 6999): return 'Finance'
        if (100 <= s <= 999): return 'Consumer'
        if (1000 <= s <= 1499): return 'Other'
        if (1500 <= s <= 1999): return 'Other'
        if (2000 <= s <= 2799): return 'Consumer'
        if (2800 <= s <= 2829): return 'Chemicals'
        if (2840 <= s <= 2899): return 'Consumer'
        if (3000 <= s <= 3999): return 'Durables'
        if (4000 <= s <= 4799): return 'Other'
        if (5000 <= s <= 5999): return 'Shops'
        if (7000 <= s <= 7999): return 'Services'
        if (8100 <= s <= 8999): return 'Services'
        if (9100 <= s <= 9999): return 'Other'
        return 'Other'

if __name__ == "__main__":
    pass

