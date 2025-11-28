import streamlit as st
import pandas as pd
import joblib
from preprocessing import CustomPreprocessor   # your class lives in preprocessing.py

# --- Load the trained pipeline ---
pipeline = joblib.load("xgboost_full_pipeline.pkl")

# --- Recursive Forecast Function ---
def recursive_forecast(pipeline, history_df, start_date, end_date):
    """
    Generate recursive forecasts between start_date and end_date.
    """
    history_df['date'] = pd.to_datetime(history_df['date'])
    df = history_df.copy().sort_values('date')

    future_dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
    preds = []

    for future_date in future_dates:
        new_row = {col: df.iloc[-1][col] for col in df.columns if col != 'date'}
        new_row['date'] = future_date

        # Dynamic features
        new_row['is_weekend'] = 1 if future_date.weekday() in [5,6] else 0
        new_row['day_of_week'] = future_date.weekday()
        new_row['month'] = future_date.month

        # Plug in holiday/promotion calendars if available
        new_row['is_holiday'] = 0
        new_row['promotion'] = 0
        new_row['is_school_in_session'] = 1

        # Append new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Placeholder target
        df.loc[df.index[-1], 'daily_sales_NGN'] = 0

        # Predict
        X_future = df.iloc[-1:].copy()
        y_pred = pipeline.predict(X_future)[0]
        preds.append(y_pred)

        # Update target with prediction
    df.loc[df.index[-1], 'daily_sales_NGN'] = y_pred

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'Predicted Sales (₦)': preds
    })
    return forecast_df

# --- Streamlit UI ---
st.title("📈 3-Month Pizza Sales Forecast")

st.write("Forecasting daily average sales from **2026-01-01 to 2026-03-31** using XGBoost.")

# Load your historical dataset (raw data from 2021-01-01 to 2025-12-31)
history_df = pd.read_csv("pizza_sales_2021_2025.csv")

# Run forecast
forecast_df = recursive_forecast(pipeline, history_df, "2026-01-01", "2026-03-31")

# Display results
st.subheader("Forecasted Sales (2026-01-01 → 2026-03-31)")
st.line_chart(forecast_df.set_index('date'))
st.dataframe(forecast_df)

# Download option
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Forecast as CSV",
    data=csv,
    file_name="sales_forecast_2026_Q1.csv",
    mime="text/csv",
)