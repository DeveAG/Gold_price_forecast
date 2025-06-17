
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Gold Forecast (v6)", layout="wide")
st.title("📈 Gold Price Forecast App (SARIMAX & Prophet)")

@st.cache_data(show_spinner=False)
def get_gold_data():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    df = yf.download('GC=F', start='2021-01-01', end=today)
    return df[['Open', 'High', 'Low', 'Close']].dropna()

@st.cache_data(show_spinner=False)
def get_intraday_data():
    df = yf.download('GC=F', period='5d', interval='1h')
    return df[['Open', 'High', 'Low', 'Close']].dropna()

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except:
        return False

def train_sarimax(df, column, future_days, status_area):
    status_area.write(f"📊 Training SARIMAX model for **{column}**...")
    model = SARIMAX(df[column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)
    pred = result.predict(start=len(df), end=len(df)+future_days-1)
    return pred

def train_prophet(df, column, future_days, status_area):
    status_area.write(f"🔮 Training Prophet model for **{column}**...")
    df_prophet = df[[column]].reset_index().rename(columns={"Date": "ds", column: "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds').iloc[-future_days:]

future_days_input = st.text_input("Enter number of days to forecast (e.g. 30):", "30")
try:
    future_days = int(future_days_input)
except ValueError:
    future_days = 30

model_choice = st.selectbox("Choose Forecasting Model", ["SARIMAX", "Prophet"])

with st.expander("🔔 Telegram Alert Setup (Optional)", expanded=False):
    telegram_token = st.text_input("Telegram Bot Token", type="password")
    chat_id = st.text_input("Telegram Chat ID")

if st.button("Run Forecast"):
    status_placeholder = st.empty()
    with st.spinner("Fetching data..."):
        df = get_gold_data()
        last_date = df.index[-1].date()
        st.success(f"Latest available price: {last_date}")

    forecasts = {}
    for col in ['Open', 'High', 'Low', 'Close']:
        if model_choice == "SARIMAX":
            forecast = train_sarimax(df, col, future_days, status_placeholder)
            forecasts[col] = forecast
        else:
            forecast = train_prophet(df, col, future_days, status_placeholder)
            forecasts[col] = forecast["yhat"]

    status_placeholder.empty()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    result_df = pd.DataFrame({col: forecasts[col].values for col in forecasts}, index=forecast_dates)
    result_df.index.name = "Forecast Date"
    st.subheader(f"📋 Forecast Table ({model_choice})")
    st.dataframe(result_df.round(2), use_container_width=True)

    csv_path = "forecast_results.csv"
    result_df.to_csv(csv_path)
    st.download_button("📥 Download Forecast as CSV", open(csv_path, "rb"), csv_path)

    if telegram_token and chat_id:
        message = f"Gold Forecast ✅ ({model_choice})\nDays: {future_days}\nLast Price Date: {last_date}"
        if send_telegram_message(telegram_token, chat_id, message):
            st.success("Telegram alert sent successfully!")
        else:
            st.warning("Failed to send Telegram alert.")

# Show intraday (1-hour) table
st.subheader("⏱️ 1-Hour Chart Table (Past 5 Days)")
intraday_df = get_intraday_data()
st.dataframe(intraday_df.tail(48).round(2), use_container_width=True)  # Show last 2 days (48 hours)
