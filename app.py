import streamlit as st
import pandas as pd
import joblib

# Load your trained pipeline
pipeline = joblib.load('xgboost_final_pipeline.pkl')

# App title
st.title("Daily Average Pizza Sales Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    input_df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:")
    st.dataframe(input_df)

    # Predict button
    if st.button("Predict"):
        predictions = pipeline.predict(input_df)
        input_df['Predicted Sales'] = predictions
        st.write("✅ Prediction Results:")
        st.dataframe(input_df)

        # Optionally download results
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
