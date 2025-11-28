from preprocessing import CustomPreprocessor
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
    # CRUCIAL: The future sales column is initialized with an average, but will be overwritten by predictions.
    full_df = pd.concat([history_df, future_exog_df], ignore_index=True)
    
    # Initialize the results list
    forecast_results = []
    
    # Variable to capture the transformed features for Day 1 debugging
    day_1_features = None
    
    # Variable to capture the relevant history for debugging
    debug_history = history_df.iloc[-7:].copy()

    # 3. Start the recursive loop
    for i in range(forecast_length):
        predict_index = history_length + i
        
        # --- A. Prepare the data slice for transformation ---
        # We need history + the date being predicted (which has the exogenous features)
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
        # The predicted value is immediately used as the input sales data for the next day's lag features.
        full_df.loc[predict_index, TARGET_COL] = prediction
        
    return pd.DataFrame(forecast_results), day_1_features, debug_history

# =================================================================
# 4. STREAMLIT APP LAYOUT
# =================================================================

# --- FIXED STYLING FUNCTION ---
def highlight_sales_lags(row):
    """Highlights rows where the Feature Name starts with 'sales_'."""
    if isinstance(row['Feature Name'], str) and row['Feature Name'].startswith('sales_'):
        # Apply yellow background to the whole row
        return ['background-color: #fffbcf'] * len(row)
    else:
        # No styling
        return [''] * len(row)

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
                forecast_df, day_1_features, debug_history = run_recursive_forecast(
                    pipeline, pd.to_datetime(start_date), FORECAST_LENGTH, last_known_sales
                )
            
            # Display Results
            st.success(f"✅ {FORECAST_LENGTH}-Day Forecast Complete. First Prediction: ₦{forecast_df.iloc[0][TARGET_COL]:,.2f}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label=f"Average Daily Sales ({FORECAST_LENGTH} Days)",
                          value=f"₦{forecast_df[TARGET_COL].mean():,.2f}")
                
                st.subheader("Raw Forecast Data")
                st.dataframe(forecast_df.head(10).style.format({TARGET_COL: "₦{:,.2f}"}), use_container_width=True)
                
            with col2:
                st.subheader(f"Sales Forecast Trend ({FORECAST_LENGTH} Days)")
                st.line_chart(forecast_df, x='date', y=TARGET_COL, use_container_width=True)
                
            st.markdown("---")

            # --- DEBUGGING STEP 1: VERIFY ANCHOR HISTORY ---
            st.subheader("🕵️ Debug Check 1: Anchoring Data")
            st.markdown("Verify the `Last Known Sales` value made it into the simulated history used for lag calculation.")
            
            st.dataframe(
                debug_history.style.format({TARGET_COL: "₦{:,.2f}"}), 
                column_config={"date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD", disabled=True)},
                use_container_width=True
            )
            st.markdown(f"**Expected Check:** The last row in the `{TARGET_COL}` column above should be **exactly** ₦{last_known_sales:,.2f}.")


            # --- DEBUGGING STEP 2: CHECK TRANSFORMED FEATURES ---
            st.subheader("🛠️ Debug Check 2: Day 1 Transformed Features")
            st.markdown(
                "Compare these features to the input data your model expects. **The yellow-highlighted sales lags are critical.**"
            )
            
            # Prepare debug DataFrame
            debug_df = pd.DataFrame({
                'Feature Name': day_1_features.columns,
                'Value': day_1_features.iloc[0].values
            })
            
            # Displaying column names and values side-by-side (using row-wise style application)
            st.dataframe(
                debug_df.style.apply(highlight_sales_lags, axis=1) # FIX APPLIED HERE
                      .format({'Value': '{:.4f}'}), 
                use_container_width=True,
                height=500
            )

            st.markdown("**Expected Check:** The `sales_lag_1` feature value in the highlighted row above should be exactly or very close to **₦95,106.00**. If it is a low number like `2000` or `0.0`, the error is in the `transform` step.")
            
            st.markdown("---")
            st.download_button(
                label="Download Full Forecast CSV",
                data=forecast_df.to_csv(index=False).encode('utf-8'),
                file_name=f'sales_forecast_{start_date.strftime("%Y%m%d")}_{FORECAST_LENGTH}d.csv',
                mime='text/csv',
            )