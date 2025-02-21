import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_sarima(train, steps, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    fit = model.fit()
    return fit.forecast(steps)