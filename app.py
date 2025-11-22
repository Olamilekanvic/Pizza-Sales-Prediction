import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta, date
# New Imports for consolidated preprocessor
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor 

# =================================================================
# 1. CONSTANTS (Must match the training script)
# =================================================================

TARGET_COL = 'daily_sales_NGN'
PIPELINE_FILE = 'xgboost_full_pipeline.pkl' 
# Reverting to False, as requested. 
PERFORM_INVERSE_TRANSFORM = False 

# --- NEW FORECAST CONSTANT ---
FORECAST_LENGTH = 90 # Predict 90 days (approx. 3 months)

# --- Preprocessor Column Definitions (Copied from former custom_preprocessor.py) ---
CAT_COLS = ['Day_Name', 'Rainfall_Category'] 
BOOL_COLS = ['is_weekend', 'is_holiday', 'is_rainy', 'is_school_in_session', 'promotion'] 


# =================================================================
# 2. CUSTOM PREPROCESSOR CLASS (CONSOLIDATED)
# =================================================================

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    A transformer that performs time-series feature engineering (lags, rolling averages)
    and one-hot/cyclical encoding.
    """
    def __init__(self, top_features, cat_cols=CAT_COLS, bool_cols=BOOL_COLS, target_col=TARGET_COL):
        # Initializing parameters passed from the Pipeline definition
        self.top_features = top_features
        self.cat_cols = cat_cols
        self.bool_cols = bool_cols
        self.target_col = target_col
        self.impute_value = None 
        self.ohe_features = None 

    def fit(self, X, y=None):
        # 1. Calculate Imputation Value (mean of the target)
        if self.target_col in X.columns:
            self.impute_value = X[self.target_col].mean()
        else:
            # Using y if the target column was passed separately
            self.impute_value = y.mean()
            
        # 2. Fit the One-Hot-Encoder to learn all categories
        temp_df = X.copy()
        valid_cat_cols = [col for col in self.cat_cols if col in temp_df.columns]
        temp_df = pd.get_dummies(temp_df, columns=valid_cat_cols, drop_first=True)
        self.ohe_features = [col for col in temp_df.columns if any(c in col for c in valid_cat_cols)]
        
        return self

    def transform(self, X):
        df = X.copy()
        
        # --- PRE-PROCESSING ---
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


# =================================================================
# 3. HELPER FUNCTIONS
# =================================================================

@st.cache_resource
def load_pipeline():
    """Loads the trained pipeline using joblib."""
    try:
        pipeline = joblib.load(PIPELINE_FILE)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Pipeline file '{PIPELINE_FILE}' not found. "
                 "Please ensure you have run the training script and saved the file.")
        return None
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

def generate_history_df(start_date, history_length, anchor_sales):
    """
    Generates a DataFrame of simulated historical data (30 days) needed 
    to calculate the initial set of lag features, anchored to the last known sales.
    """
    history_dates = [start_date - timedelta(days=i) for i in range(history_length, 0, -1)]
    
    # Calculate a slight variance range around the anchor sales value (e.g., +/- 10%)
    lower_bound = anchor_sales * 0.90
    upper_bound = anchor_sales * 1.10
    
    # Generate sales data centered around the anchor
    simulated_sales = np.random.uniform(lower_bound, upper_bound, history_length)
    
    # The last day of the history MUST match the anchor exactly for sales_lag_1 accuracy
    simulated_sales[-1] = anchor_sales
    
    history_df = pd.DataFrame({
        'date': history_dates,
        TARGET_COL: simulated_sales, 
        # Using realistic dummy values for other features
        'transactions_count': np.random.randint(100, 200, history_length),
        'avg_order_value_NGN': np.random.uniform(500, 700, history_length),
        'foot_traffic_index': np.random.rand(history_length) * 8,
        'temperature_C': np.random.rand(history_length) * 30,
        # Default/neutral values for boolean/categorical columns for history
        'Day_Name': [d.strftime('%a') for d in history_dates],
        'Rainfall_Category': np.random.choice(['None'], history_length),
        'is_weekend': [1 if d.dayofweek >= 5 else 0 for d in history_dates],
        'is_holiday': 0,
        'is_rainy': 0,
        'is_school_in_session': 1,
        'promotion': 0,
    })
    
    return history_df

def create_future_exogenous_features(start_date, forecast_length, history_df):
    """
    Creates a DataFrame of exogenous features for the entire forecast period.
    """
    future_dates = [start_date + timedelta(days=i) for i in range(forecast_length)]
    
    # Calculate simple means from the anchored history for unknown future features
    mean_transactions = history_df['transactions_count'].mean()
    mean_aov = history_df['avg_order_value_NGN'].mean()
    mean_traffic = history_df['foot_traffic_index'].mean()
    mean_temp = history_df['temperature_C'].mean()

    future_df = pd.DataFrame({
        'date': future_dates,
        # --- Time-dependent non-sales features (can be derived from date) ---
        'Day_Name': [d.strftime('%a') for d in future_dates],
        'is_weekend': [1 if d.dayofweek >= 5 else 0 for d in future_dates],
        # --- Time-INDEPENDENT, UNKNOWN future features (using averages/assumptions) ---
        'transactions_count': mean_transactions,
        'avg_order_value_NGN': mean_aov,
        'foot_traffic_index': mean_traffic,
        'temperature_C': mean_temp,
        # --- UNKNOWN features (using neutral/default assumptions) ---
        'Rainfall_Category': 'None', 
        'is_holiday': 0,
        'is_rainy': 0,
        'is_school_in_session': 1, # Assume schools are generally in session
        'promotion': 0, # Assume no promotion unless specified by the user/schedule
    })
    
    return future_df

def run_recursive_forecast(pipeline, start_date, forecast_length, last_known_sales):
    """
    Performs the multi-step (recursive) forecast using an anchored history.
    """
    history_length = 30
    
    # 1. Get initial history (anchored to the user's last known sales value)
    history_df = generate_history_df(start_date, history_length, last_known_sales)
    
    # 2. Generate exogenous features for the 90-day forecast period
    future_exog_df = create_future_exogenous_features(start_date, forecast_length, history_df)
    
    # Combine history and the future exogenous data to form the full prediction frame
    full_df = pd.concat([history_df, future_exog_df], ignore_index=True)
    
    # Initialize the results list
    forecast_results = []
    
    # Variable to capture the transformed features for Day 1 debugging
    day_1_features = None

    # 3. Start the recursive loop
    for i in range(forecast_length):
        predict_index = history_length + i
        
        # --- A. Prepare the data slice for transformation ---
        X_predict_slice = full_df.iloc[:predict_index + 1].copy()
        
        # --- B. Transform the slice ---
        X_transformed = pipeline['preprocessing'].transform(X_predict_slice)
        
        # --- C. Select the feature vector for the prediction date (last row) ---
        X_final_predict = X_transformed.iloc[[-1]]

        # --- D. Capture Day 1 features for debugging ---
        if i == 0:
            day_1_features = X_final_predict.copy()

        # --- E. Predict ---
        prediction = pipeline['model'].predict(X_final_predict)[0]
        
        # --- F. Store Result ---
        forecast_date = X_predict_slice.iloc[-1]['date']
        forecast_results.append({
            'date': forecast_date,
            TARGET_COL: prediction
        })
        
        # --- G. CRITICAL RECURSIVE STEP: Update the full_df with the prediction ---
        full_df.loc[predict_index, TARGET_COL] = prediction
        
    return pd.DataFrame(forecast_results), day_1_features

# =================================================================
# 4. STREAMLIT APP LAYOUT
# =================================================================

st.set_page_config(page_title="90-Day Sales Forecast", layout="wide")
st.title("90-Day Daily Sales Forecast Predictor (Recursive Model)")
st.caption(f"This app performs a multi-step forecast for the next {FORECAST_LENGTH} days.")

# Load the pipeline
pipeline = load_pipeline()

if pipeline:
    st.sidebar.header("Forecast Settings")
    
    # UI input for the anchoring value
    last_known_sales = st.sidebar.number_input(
        "Last Known Daily Sales Value (NGN)", 
        value=95106.00, 
        min_value=1.0, 
        step=100.0,
        format="%.2f",
        help="Input the last actual sales value from your dataset (e.g., the last day of Sept 2025). This anchors the forecast."
    )
    
    # Use today as the start date for the forecast
    start_date = st.sidebar.date_input("Start Date (Day 1 of Forecast)", value=date.today() + timedelta(days=1))
    
    st.sidebar.info(f"The model will forecast {FORECAST_LENGTH} days from {start_date.strftime('%Y-%m-%d')}.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Forecasting Assumptions")
    st.sidebar.warning(
        "For the 90-day forecast, unknown features (e.g., transactions, temperature, promotions) "
        "are assumed to be their historical average values. **Accuracy depends on the quality of these assumptions.**"
    )

    if st.button(f"Run {FORECAST_LENGTH}-Day Forecast", type="primary"):
        
        if last_known_sales < 5000:
             st.error("Please enter a realistic 'Last Known Daily Sales Value' to anchor the forecast.")
        else:
            with st.spinner(f'Running recursive forecast for {FORECAST_LENGTH} days...'):
                forecast_df, day_1_features = run_recursive_forecast(pipeline, pd.to_datetime(start_date), FORECAST_LENGTH, last_known_sales)
            
            # Display Results
            st.success(f"✅ {FORECAST_LENGTH}-Day Forecast Complete")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                avg_sales = forecast_df[TARGET_COL].mean()
                st.metric(label=f"Average Daily Sales ({FORECAST_LENGTH} Days)",
                          value=f"₦{avg_sales:,.2f}")
                st.metric(label="Total Projected Sales",
                          value=f"₦{forecast_df[TARGET_COL].sum():,.2f}")
                
                st.subheader("Raw Forecast Data")
                st.dataframe(forecast_df.head(10).style.format({TARGET_COL: "₦{:,.2f}"}), use_container_width=True)
                
                st.markdown("---")
                st.download_button(
                    label="Download Full Forecast CSV",
                    data=forecast_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'sales_forecast_{start_date.strftime("%Y%m%d")}_{FORECAST_LENGTH}d.csv',
                    mime='text/csv',
                )
                
            with col2:
                st.subheader(f"Sales Forecast Trend ({FORECAST_LENGTH} Days)")
                st.line_chart(forecast_df, x='date', y=TARGET_COL, use_container_width=True)
                
            # --- DEBUGGING OUTPUT ---
            st.subheader("🛠️ Debugging: Day 1 Input Feature Vector")
            st.warning(
                "This table shows the exact features and values the model received for its first prediction. "
                "Compare these columns and values to the features used in your **training script**."
            )
            
            # Displaying column names and values side-by-side
            debug_df = pd.DataFrame({
                'Feature Name': day_1_features.columns,
                'Value': day_1_features.iloc[0].values
            })
            
            # Format the output for readability
            st.dataframe(
                debug_df.style.format({'Value': '{:.4f}'}), 
                use_container_width=True,
                height=400
            )

            st.markdown("---")
            st.subheader("💡 Understanding Recursive Forecasting")
            st.write(
                "For a multi-step forecast, the model uses its own previous predictions to generate "
                "the lagged features needed for the next day. This process compounds any errors, "
                "making the near-term predictions more reliable than the far-term ones."
            )