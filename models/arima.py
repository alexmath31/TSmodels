import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import itertools

def differencing_order(data, max_d=2):
    d = 0
    temp_data = data.copy()

    while d < max_d:
        adf_test = adfuller(temp_data)
        if adf_test[1] <= 0.05: 
            break
        d += 1
        temp_data = temp_data.diff().dropna()

    return d

def analyze_acf_pacf(data):

    acf_values = np.abs(acf(data, nlags=20, fft=True))
    pacf_values = np.abs(pacf(data, nlags=20))

    p_guess = np.sum(pacf_values[1:] > 0.2)  
    q_guess = np.sum(acf_values[1:] > 0.2)

    return min(p_guess, 3), min(q_guess, 3)

def forecast_arima(train, test, order=None):

    d = differencing_order(train)
    differenced_train = train.diff(d).dropna() if d > 0 else train

    if order is None:
        p, q = analyze_acf_pacf(differenced_train)
        order = (p, d, q)

    model = ARIMA(train, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.forecast(len(test))

    return forecast

