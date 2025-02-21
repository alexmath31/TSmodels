import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def plot_distribution(data, title=""):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data, kde=True, ax=ax[0])
        ax[0].set_title(f"Histogram & KDE - {title}")
        sns.boxplot(data=data, ax=ax[1])
        ax[1].set_title(f"Boxplot - {title}")
        st.pyplot(fig)

def show_overview(df):
    st.write("### Time Series Plot:")
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    st.pyplot(fig)

    st.write("### Initial Data Distribution")
    plot_distribution(df, "Original Data")

    st.write("### Seasonality Detection")
    lags = 30  
    autocorr_values = acf(df.dropna(), nlags=lags)
    peak_lag = np.argmax(autocorr_values[1:]) + 1 

    if autocorr_values[peak_lag] > 0.5:
        st.write(f"Possible seasonal pattern detected with period: **{peak_lag}**")
    else:
        st.write("No strong seasonality detected.")

    st.write("### Data Transformation")
    transformation = st.selectbox("Choose transformation:", ["None", "Log Transform", "Differencing", "Box-Cox"])

    if transformation == "Log Transform":
        df = np.log1p(df)
        st.write("Applied **Log Transform**")
    elif transformation == "Differencing":
        df = df.diff().dropna()
        st.write("Applied **Differencing**")
    elif transformation == "Box-Cox":
        epsilon = 1e-6 - df.min().min() if df.min().min() <= 0 else 0
        df_positive = df + epsilon  

        df_boxcox = pd.DataFrame()  
        for col in df_positive.columns:
            df_boxcox[col], _ = stats.boxcox(df_positive[col].dropna())  

        df = df_boxcox
        st.write("Applied **Box-Cox Transformation**")

    if transformation != "None":
        st.write("### Data Distribution After Transformation")
        plot_distribution(df, "After Transformation")

    st.write("### Normalize Data")
    normalization_method = st.selectbox("Choose normalization method:", ["None", "StandardScaler", "MinMaxScaler"])

    if normalization_method == "StandardScaler":
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        st.write("Data normalized using **StandardScaler**")
    elif normalization_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        st.write("Data normalized using **MinMaxScaler**")

    if normalization_method != "None":
        st.write("### Data Distribution After Normalization")
        plot_distribution(df, "After Normalization")

    st.write("### Processed Data Overview:")
    st.write(df.describe())

    st.write("### Missing Values After Preprocessing:")
    st.write(df.isnull().sum())

    return df
