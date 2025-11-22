## Streamlit App for Daily Pizza Sales Prediction using Lag-Based XGBoost Model

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta, date
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor # Keep this import, even if the model is loaded

# =================================================================
# 1. CONSTANTS (Must match the training script)
# =================================================================

TARGET_COL = 'daily_sales_NGN'
CAT_COLS = ['day_of_week', 'month', 'public_holiday_name', 'university_calendar_status', 'weather'] 
BOOL_COLS = ['is_weekend', 'is_holiday', 'is_school_in_session', 'promotion'] 
PIPELINE_FILE = 'xgboost_full_pipeline.pkl' # Name of the file saved by joblib

# =================================================================
# 2. CUSTOM PREPROCESSOR CLASS (Must be redefined for joblib loading)
# =================================================================

# NOTE: This class must be identical to the one used during training 
# so that joblib can successfully load the pipeline.

class CustomPreprocessor(BaseEstimator, TransformerMixin):
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
        # Crucially, this must run on the combined input data (Day N-30 to Day N)
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
        

        # --- Imputation (Fills NaNs from lags/rolling) ---
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
             for col in missing:
                  df[col] = 0
                  
        return df[self.top_features]


# =================================================================
# 3. HELPER FUNCTIONS
# =================================================================

@st.cache_resource
def load_pipeline():
    """Loads the trained pipeline using joblib."""
    try:
        # Load the pipeline object
        pipeline = joblib.load(PIPELINE_FILE)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Pipeline file '{PIPELINE_FILE}' not found. "
                 "Please ensure you have run the training script and saved the file.")
        return None
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

def create_prediction_df(input_data, pipeline):
    """
    Creates the DataFrame required by the CustomPreprocessor for prediction.
    
    This function simulates retrieving the previous 30 days of historical data (dummy_history)
    and combines it with the current day's data (input_data) to allow the lag and
    rolling features to be calculated correctly.
    """
    prediction_date = input_data['date']

    # --- 1. Simulate Historical Data (Crucial for Lags/Rolling) ---
    # In a real app, you would fetch this from a database. 
    history_length = 30
    
    # Create dates for the 30 days prior to the prediction date
    history_dates = [prediction_date - timedelta(days=i) for i in range(history_length, 0, -1)]
    
    # Create dummy sales and traffic data for the history
    # NOTE: In a REAL application, you MUST fetch the actual sales, transactions, and traffic 
    # data for these 30 days from your database/data source.
    history_df = pd.DataFrame({
        'date': history_dates,
        TARGET_COL: np.random.uniform(60000, 10000, history_length), # DUMMY historical sales
        'transactions_count': np.random.randint(50, 150, history_length),
        'foot_traffic_index': np.random.rand(history_length) * 8,
    })
    
    # Fill in required categorical/boolean columns with neutral/default values for history
    for col in CAT_COLS:
        history_df[col] = history_df['date'].dt.day_name().str[:3] # Example default (e.g., Mon)
    for col in BOOL_COLS:
        history_df[col] = 0
    history_df['temperature_C'] = np.random.rand(history_length) * 30
    history_df['avg_order_value_NGN'] = np.random.rand(history_length) * 1000

    # --- 2. Create Current Day Data ---
    current_day_df = pd.DataFrame({
        'date': [input_data['date']],
        'transactions_count': [input_data['transactions_count']],
        'avg_order_value_NGN': [input_data['avg_order_value_NGN']],
        'foot_traffic_index': [input_data['foot_traffic_index']],
        'temperature_C': [input_data['temperature_C']],
        'Day_Name': [input_data['Day_Name']],
        'Rainfall_Category': [input_data['Rainfall_Category']],
        'is_weekend': [input_data['is_weekend']],
        'is_holiday': [input_data['is_holiday']],
        'is_rainy': [input_data['is_rainy']],
        'is_school_in_session': [input_data['is_school_in_session']],
        'promotion': [input_data['promotion']],
        # IMPORTANT: The target column is required for the transformer to calculate lags.
        # We use the training mean as a placeholder for the day we are predicting.
        TARGET_COL: [pipeline['preprocessing'].impute_value]
    })
    
    # --- 3. Combine and Return ---
    # Combine history and the current day (31 rows total).
    combined_df = pd.concat([history_df, current_day_df], ignore_index=True)

    return combined_df

# =================================================================
# 4. STREAMLIT APP LAYOUT
# =================================================================

st.set_page_config(page_title="Sales Forecasting App", layout="centered")
st.title("Daily Sales Forecast Predictor (Time-Series Model)")
st.caption("This app uses a lag-based XGBoost model. Inputs are collected to forecast a single future day's sales.")

# Load the pipeline
pipeline = load_pipeline()

if pipeline:
    st.sidebar.header("Prediction Date & Inputs")
    
    # --- 4.1. User Input Fields ---
    
    # Use today as default date, but allow future dates
    prediction_date = st.sidebar.date_input("Select Prediction Date", value=date.today() + timedelta(days=1))
    
    # Map prediction date to features
    dt = pd.to_datetime(prediction_date)
    day_name = dt.strftime('%a')
    
    st.sidebar.markdown(f"**Predicted Day Name:** `{day_name}`")
    
    # Input numerical features
    transactions_count = st.sidebar.number_input("Expected Transactions Count", min_value=1, value=100)
    avg_order_value_NGN = st.sidebar.number_input("Average Order Value (NGN)", min_value=10.0, value=500.0)
    foot_traffic_index = st.sidebar.number_input("Expected Foot Traffic Index", min_value=0.0, value=5.0)
    temperature_C = st.sidebar.number_input("Expected Temperature (°C)", min_value=0.0, value=25.0)

    # Input categorical/boolean features
    rainfall_category = st.sidebar.selectbox("Rainfall Category", options=['None', 'Light', 'Heavy'], index=0)
    promotion = st.sidebar.radio("Is there a promotion today?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
    is_holiday = st.sidebar.radio("Is it a Public Holiday?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
    is_school_in_session = st.sidebar.radio("Are schools in session?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)

    # Calculate boolean features based on day
    is_weekend = 1 if dt.dayofweek >= 5 else 0 # Saturday (5) or Sunday (6)
    is_rainy = 1 if rainfall_category != 'None' else 0 # Derived from category

    # Display derived features
    st.sidebar.markdown("---")
    st.sidebar.info(f"Derived Features:\n- Weekend: {'Yes' if is_weekend else 'No'}\n- Rainy: {'Yes' if is_rainy else 'No'}")


    # --- 4.2. Prediction Logic ---
    
    if st.button("Calculate Forecast"):
        
        # 1. Gather all inputs into a dictionary
        input_data = {
            'date': dt,
            'transactions_count': transactions_count,
            'avg_order_value_NGN': avg_order_value_NGN,
            'foot_traffic_index': foot_traffic_index,
            'temperature_C': temperature_C,
            'Day_Name': day_name,
            'Rainfall_Category': rainfall_category,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'is_rainy': is_rainy,
            'is_school_in_session': is_school_in_session,
            'promotion': promotion,
        }
        
        # 2. Create the full prediction DataFrame (31 days: 30 history + 1 prediction day)
        with st.spinner('Preparing features and fetching historical context...'):
            full_df = create_prediction_df(input_data, pipeline)
            
            # The model only needs the features for the LAST day (the prediction day)
            # but the transformer must run on the full 31-day history.
            X_predict_full = full_df.copy()

            # 3. Transform the data (this calculates all lags and OHE on the 31 rows)
            X_transformed_full = pipeline['preprocessing'].transform(X_predict_full)
            
            # 4. Select only the features for the final prediction date (the last row)
            X_final_predict = X_transformed_full.iloc[[-1]]
        
        # 5. Predict
        prediction = pipeline['model'].predict(X_final_predict)[0]
        
        # 6. Display Result
        st.success("✅ Forecast Complete")
        st.metric(label=f"Predicted Sales for {prediction_date.strftime('%Y-%m-%d')}",
                  value=f"₦{prediction:,.2f}")

        st.subheader("⚠️ Important Note on Historical Data")
        st.warning(
            "The model used **simulated** historical sales and traffic data for the last 30 days "
            "to calculate the necessary lagged features. For production use, the `create_prediction_df` "
            "function MUST be updated to fetch **actual** historical values from your database."
        )
        
