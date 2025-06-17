import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Gold Forecast", layout="wide")
st.title("📈 Gold Price Forecast App")

# --------------- Data Load ------------------

@st.cache_data(show_spinner=False)
def get_gold_data_daily():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    df = yf.download('GC=F', start='2021-01-01', end=today)
    return df[['Open', 'High', 'Low', 'Close']].dropna()

@st.cache_data(show_spinner=False)
def get_gold_data_intraday():
    df = yf.download('GC=F', period='7d', interval='1h')
    return df[['Open', 'High', 'Low', 'Close']].dropna()

# --------------- Forecast Models ------------------

def train_sarimax(df, column, steps, freq, status_area):
    status_area.write(f"📊 SARIMAX training for {column} ({freq})...")
    model = SARIMAX(df[column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12 if freq == 'D' else 24))
    result = model.fit(disp=False)
    pred = result.forecast(steps)
    return pred

def train_prophet(df, column, steps, freq, status_area):
    status_area.write(f"🔮 Prophet training for {column} ({freq})...")
    df = df[[column]].reset_index()
    df.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=(freq == 'D'))
    if freq == 'H':
        model.add_seasonality(name='hourly', period=24, fourier_order=3)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds').iloc[-steps:]

# --------------- Telegram (Optional) ------------------

def send_telegram_message(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)
    except:
        pass

# --------------- User Inputs ------------------

col1, col2 = st.columns(2)
with col1:
    future_days = st.number_input("🔢 Forecast Days (daily)", min_value=1, value=30)
with col2:
    future_hours = st.number_input("⏱️ Forecast Hours (1H)", min_value=1, value=24)

model_choice = st.radio("📌 Select Forecasting Model", ["SARIMAX", "Prophet"])

with st.expander("🔔 Telegram Alert Setup (Optional)", expanded=False):
    telegram_token = st.text_input("Telegram Bot Token", type="password")
    chat_id = st.text_input("Telegram Chat ID")

# --------------- Run Button ------------------

if st.button("📊 Run Forecast"):
    status = st.empty()
    with st.spinner("Fetching price data..."):
        df_daily = get_gold_data_daily()
        df_hourly = get_gold_data_intraday()

    last_day = df_daily.index[-1]
    last_hour = df_hourly.index[-1]

    daily_result = pd.DataFrame()
    hourly_result = pd.DataFrame()

    # --- Daily Forecast ---
    try:
        daily_forecasts = {}
        for col in df_daily.columns:
            if model_choice == "SARIMAX":
                daily_forecasts[col] = train_sarimax(df_daily, col, future_days, "D", status)
            else:
                forecast = train_prophet(df_daily, col, future_days, "D", status)
                daily_forecasts[col] = forecast["yhat"]

        daily_result = pd.DataFrame(daily_forecasts)
        daily_result.index = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=future_days, freq='D')
        daily_result.index.name = "Date"
    except Exception as e:
        st.warning(f"⚠️ Daily forecast failed: {e}")

    # --- 1H Forecast ---
    try:
        hourly_forecasts = {}
        for col in df_hourly.columns:
            if model_choice == "SARIMAX":
                hourly_forecasts[col] = train_sarimax(df_hourly, col, future_hours, "H", status)
            else:
                forecast = train_prophet(df_hourly, col, future_hours, "H", status)
                hourly_forecasts[col] = forecast["yhat"]

        hourly_result = pd.DataFrame(hourly_forecasts)
        hourly_result.index = pd.date_range(start=last_hour + pd.Timedelta(hours=1), periods=future_hours, freq='H')
        hourly_result.index.name = "DateTime"
    except Exception as e:
        st.warning(f"⚠️ Hourly forecast failed: {e}")

    status.empty()

    # --- Display Forecast Tables ---
    if not daily_result.empty:
        st.subheader("📅 Daily Forecast Table")
        st.dataframe(daily_result.round(2), use_container_width=True)

    if not hourly_result.empty:
        st.subheader("⏱️ 1-Hour Forecast Table")
        st.dataframe(hourly_result.round(2), use_container_width=True)

    # --- Telegram Alert ---
    if telegram_token and chat_id:
        msg = f"✅ Gold Forecast Complete!\nModel: {model_choice}\nForecasted: {future_days}d & {future_hours}h"
        send_telegram_message(telegram_token, chat_id, msg)
        st.success("Telegram notification sent.")
<full app code here>
