import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.arima import forecast_arima
from models.sarima import forecast_sarima
from models.lstm import forecast_lstm
from models.neural_net import forecast_nn
from models.kan import forecast_kan
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

def show_forecast(df, transformation="Differencing", normalization="StandardScaler"):
    
    train_size = int(len(df) * 0.85)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    st.write("### Choose Forecasting Methods:")
    methods = st.multiselect(
        "Select one or more methods:",
        ["LSTM", "ARIMA", "SARIMA", "Neural Network", "Kolmogorov-Arnold Network"]
    )

    if not methods:
        st.warning("Please select at least one forecasting method.")
        return

    forecast_results = {}
    errors = {}

    steps = len(test)

    for method in methods:
        match method:
            case "ARIMA":
                forecast_results["ARIMA"] = forecast_arima(train, steps)
            case "SARIMA":
                forecast_results["SARIMA"] = forecast_sarima(train, steps)
            case "LSTM":
                forecast_results["LSTM"] = forecast_lstm(train, steps)
            case "Neural Network":
                forecast_results["Neural Network"] = forecast_nn(train, steps)
            case "Kolmogorov-Arnold Network":
                forecast_results["Kolmogorov-Arnold Network"] = forecast_kan(train, steps)
    
    st.write("### Forecasted Time Series")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(ax=ax, label="Original Data", color="black", linewidth=2)

    for method, forecast in forecast_results.items():
        forecast.plot(ax=ax, label=method, linestyle="--")

    ax.legend()
    st.pyplot(fig)

    st.write("### Model Performance Comparison")
    st.write(errors)
    best_model = min(errors, key=lambda k: errors[k]["RMSE"], default=None)
    if best_model:
        st.success(f"Best model: **{best_model}**")
