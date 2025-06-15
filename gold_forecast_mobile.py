import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

import warnings
warnings.filterwarnings("ignore")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true, y_pred):
    return {
        'RMSE': sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }

def get_gold_data():
    df = yf.download('GC=F', start='2018-01-01')
    df = df[['Close']].dropna()
    df.columns = ['Price']
    df.index.name = 'Date'
    return df

def train_test_split(data, test_size=0.1):
    split = int(len(data) * (1 - test_size))
    return data[:split], data[split:]

def sarimax_forecast(train, test, future_days):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)
    test_pred = result.predict(start=len(train), end=len(train)+len(test)-1)
    future_pred = result.predict(start=len(train)+len(test), end=len(train)+len(test)+future_days-1)
    return test_pred, future_pred

def prophet_forecast(train_df, test_df, future_days):
    prophet_train = train_df.reset_index().rename(columns={'Date': 'ds', 'Price': 'y'})
    model = Prophet()
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=len(test_df) + future_days)
    forecast = model.predict(future)
    test_pred = forecast.set_index('ds').iloc[-(len(test_df) + future_days):-future_days]
    future_pred = forecast.set_index('ds').iloc[-future_days:]
    return test_pred['yhat'], future_pred['yhat']

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except Exception as e:
        return False

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📱 Gold Price Forecasting (Mobile Optimized)")
st.markdown("Forecast Gold Prices using SARIMAX and Prophet models. Optionally receive a Telegram alert.")

future_days = st.slider("Forecast days into future:", 7, 90, 30)

with st.expander("🔔 Telegram Alert Setup (Optional)", expanded=False):
    telegram_token = st.text_input("Your Telegram Bot Token", type="password")
    chat_id = st.text_input("Your Telegram Chat ID")

if st.button("📈 Run Forecast"):
    with st.spinner("Running forecasting pipeline..."):
        data = get_gold_data()
        train, test = train_test_split(data)

        sarimax_test, sarimax_future = sarimax_forecast(train['Price'], test['Price'], future_days)
        prophet_test, prophet_future = prophet_forecast(train, test, future_days)

        sarimax_metrics = evaluate(test['Price'], sarimax_test)
        prophet_metrics = evaluate(test['Price'], prophet_test)

        st.subheader("📊 Evaluation on Test Set")
        metrics_df = pd.DataFrame({
            'SARIMAX': sarimax_metrics,
            'Prophet': prophet_metrics
        }).T.round(3)
        st.dataframe(metrics_df, use_container_width=True)

        best_model = metrics_df['RMSE'].idxmin()
        st.success(f"✅ Best model: **{best_model}**")

        st.subheader("📉 Test Set Predictions")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(test.index, test['Price'], label='Actual', color='black')
        ax.plot(test.index, sarimax_test, label='SARIMAX')
        ax.plot(test.index, prophet_test, label='Prophet')
        ax.set_title("Test Set Comparison")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)

        st.subheader(f"🔮 {future_days}-Day Forecast")
        future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(data.index[-100:], data['Price'].iloc[-100:], label='Historical', color='black')
        ax2.plot(future_index, sarimax_future, label='SARIMAX Forecast')
        ax2.plot(future_index, prophet_future, label='Prophet Forecast')
        ax2.set_title("Future Forecast")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2, use_container_width=True)

        if telegram_token and chat_id:
            alert_message = f"Forecast Complete ✅\nBest model: {best_model}\nRMSE: {metrics_df.loc[best_model, 'RMSE']}"
            if send_telegram_message(telegram_token, chat_id, alert_message):
                st.success("Telegram alert sent successfully!")
            else:
                st.warning("Failed to send Telegram alert. Check token/chat ID.")
