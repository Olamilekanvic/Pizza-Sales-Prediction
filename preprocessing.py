import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



TARGET_COL = 'daily_sales_NGN'
CAT_COLS = ['day_of_week', 'month', 'public_holiday_name', 'university_calendar_status', 'weather']
BOOL_COLS = ['is_weekend', 'is_holiday', 'is_school_in_session', 'promotion'] 


# =================================================================
# 2. CUSTOM PREPROCESSOR CLASS
# =================================================================

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    A transformer that performs time-series feature engineering (lags, rolling averages)
    and one-hot/cyclical encoding.
    """
    # FIX: Ensure all required parameters are accepted here
    def __init__(self, top_features, cat_cols=CAT_COLS, bool_cols=BOOL_COLS, target_col=TARGET_COL):
        self.top_features = top_features
        self.cat_cols = cat_cols
        self.bool_cols = bool_cols
        self.target_col = target_col
        self.impute_value = None 
        self.ohe_features = None 

    def fit(self, X, y=None):
        # 1. Calculate Imputation Value
        if self.target_col in X.columns:
            self.impute_value = X[self.target_col].mean()
        else:
            self.impute_value = y.mean()
            
        # 2. Fit the One-Hot-Encoder to learn all categories (for consistency)
        temp_df = X.copy()
        valid_cat_cols = [col for col in self.cat_cols if col in temp_df.columns]
        temp_df = pd.get_dummies(temp_df, columns=valid_cat_cols, drop_first=True)
        self.ohe_features = [col for col in temp_df.columns if any(c in col for c in valid_cat_cols)]
        
        return self

    def transform(self, X):
        df = X.copy()
        
        # --- PRE-PROCESSING & INITIAL COLUMN CHECKS ---
        df['date'] = pd.to_datetime(df['date'])
        
        # --- Time-Series Features (LAG & ROLLING) ---
        if self.target_col in df.columns:
            # Sales-based features (require target)
            df['sales_lag_1'] = df[self.target_col].shift(1)
            df['sales_lag_7'] = df[self.target_col].shift(7)
            df['sales_lag_30'] = df[self.target_col].shift(30)
            df['sales_7d_avg'] = df[self.target_col].rolling(7, min_periods=1).mean()
            df['sales_30d_avg'] = df[self.target_col].rolling(30, min_periods=1).mean()

        # Non-sales based features
        df['traffic_lag_1'] = df['foot_traffic_index'].shift(1)
        df['traffic_lag_7'] = df['foot_traffic_index'].shift(7)
        df['transactions_lag_1'] = df['transactions_count'].shift(1)
        df['transactions_lag_7'] = df['transactions_count'].shift(7)
        df['traffic_7d_avg'] = df['foot_traffic_index'].rolling(7, min_periods=1).mean()
        df['transactions_7d_avg'] = df['transactions_count'].rolling(7, min_periods=1).mean()

        # --- Cyclical Time Encoding ---
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month'] = df['date'].dt.month

        # --- One-Hot Encoding ---
        valid_cat_cols = [col for col in self.cat_cols if col in df.columns]
        df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True)
        
        # Add missing OHE columns (set to 0) to maintain feature set consistency
        if self.ohe_features:
            for feature in self.ohe_features:
                if feature not in df.columns:
                    df[feature] = 0

        # --- Encode boolean columns as integers ---
        for col in self.bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # --- Interactive Features ---
        df['traffic_promo'] = df['foot_traffic_index'] * df['promotion']
        df['holiday_promo'] = df['is_holiday'] * df['promotion']
        df['weekend_promo'] = df['is_weekend'] * df['promotion']
        df['school_promo'] = df['is_school_in_session'] * df['promotion']
        df['holiday_traffic'] = df['is_holiday'] * df['foot_traffic_index']
        

        # --- Imputation (Fills NaNs from lags/rolling with the learned training mean) ---
        if self.impute_value is not None:
             df = df.fillna(self.impute_value)
        else:
             df = df.fillna(0) 

        # --- Drop columns not needed for modeling ---
        cols_to_drop = ['date', 'day_of_week', 'month']
        if self.target_col in df.columns:
             cols_to_drop.append(self.target_col)
             
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # --- Select Top Features (Ensures consistent feature order and size) ---
        missing = [col for col in self.top_features if col not in df.columns]
        if missing:
             # Fill missing columns (often OHE features not present in the current slice) with 0
             for col in missing:
                  df[col] = 0
                  
        return df[self.top_features]
