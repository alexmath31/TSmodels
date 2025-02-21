import streamlit as st
import pandas as pd
from openai import OpenAI
from buttons.feature_selection import select_feature
from buttons.overview import show_overview
from buttons.forecast import show_forecast

OPENAI_API_KEY = ""

client = OpenAI(api_key=OPENAI_API_KEY)

def get_forecasting_model_info(model_name):
    
    prompt = f"""
    Explain the forecasting model '{model_name}' in simple terms, focusing on time series analysis. 
    Include a brief description of how it works, its ideal use cases, pros, and cons.
    Keep the response short and concise.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert in time series forecasting. Provide short and concise responses."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def get_time_series_description(df):
    prompt = f"""
    Analyze the given time series data and describe what it might represent. 
    Identify possible trends, seasonality, and anomalies. 
    Keep the explanation concise and intuitive for a non-expert.
    Keep explanation short.
    Here are the first few rows of the dataset:
    {df.head().to_string()}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert data analyst."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    st.title("Time Series Data Submission")
    
    uploaded_file = st.file_uploader("Upload your time series CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0, sep=r'[,|;\t"]+(?=\S)')
        
        if not isinstance(df.index, pd.DatetimeIndex):
            possible_time_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            
            if possible_time_cols:
                df[possible_time_cols[0]] = pd.to_datetime(df[possible_time_cols[0]], errors="coerce")
                df.set_index(possible_time_cols[0], inplace=True)
        
        st.write("### Processed Data:")
        st.write(df.head())

        if "ts_description" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("Analyzing dataset..."):
                st.session_state.ts_description = get_time_series_description(df)
            st.session_state.uploaded_file_name = uploaded_file.name 

        st.write("### Time Series Insights")
        st.write(st.session_state.ts_description)   

        selected_feature = select_feature(df)
        if selected_feature:
            st.write(f"**Selected Feature:** {selected_feature}")
            ts_df = df[[selected_feature]] 

            if "show_section" not in st.session_state:
                st.session_state.show_section = None

            if st.button("Show Overview"):
                st.session_state.show_section = "overview"
            
            if st.session_state.show_section == "overview":
                ts_df = show_overview(ts_df)

                if st.button("Make Forecasts"):
                    st.session_state.show_section = "forecast"

            if st.session_state.show_section == "forecast":
                show_forecast(ts_df)

                st.write("## Choose a Forecasting Model")
                forecasting_model = st.selectbox("Select a forecasting method:", ["LSTM", "ARIMA", "SARIMA", "Neural Network", "Kolmogorov-Arnold Network"])

                if forecasting_model:
                    st.write(f"### {forecasting_model} Model Details")
                    
                    with st.spinner("Fetching model insights..."):
                        model_info = get_forecasting_model_info(forecasting_model)
                    
                    st.write(model_info)

if __name__ == "__main__":
    main()