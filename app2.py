# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("xgboost_model.pkl")
preprocessor = CustomPreprocessor(top_features=[...])  # insert your top features

st.title("📈 Daily Sales Forecast Dashboard")

uploaded_file = st.file_uploader("Upload new data (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = preprocessor.transform(df)
    X_clean = pd.DataFrame(X).dropna()
    preds = model.predict(X_clean)

    st.subheader("📊 Forecasted Sales")
    st.line_chart(preds)

    st.subheader("📥 Download Predictions")
    output = pd.DataFrame({'Predicted Sales (₦)': preds})
    st.dataframe(output)