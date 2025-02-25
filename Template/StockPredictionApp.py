import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Define Start Date
START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit App Title
st.title("📈 Stock Prediction App")

# Stock Selection Dropdown
stocks = ("AAPL", "GOOG", "GME", "TSLA", "AMZN", "NVDA", "META", "MSFT")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Slider for selecting Forecast Period
n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365  # Convert years to days

# Function to load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)

    # Fix MultiIndex issue
    if isinstance(data.columns, pd.MultiIndex):
        data = data[[('Date', ''), ('Close', ticker)]].copy()
        data.columns = ['Date', 'Close']  # Rename columns

    return data

# Load data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Check if 'Close' column exists
if 'Close' in data.columns:
    data.dropna(subset=['Close'], inplace=True)
else:
    st.error("Column 'Close' is missing in stock data. Cannot proceed.")
    st.stop()

# Show raw data
st.subheader("📊 Raw Data")
st.write(data.tail())

# Plot historical stock prices
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='green')))
    fig.update_layout(title_text="📅 Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Prepare Data for Forecasting
df_train = data[['Date', 'Close']].copy()

# Rename columns for Prophet
df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Remove NaN values in y column
df_train.dropna(inplace=True)

# Ensure y column is numeric
df_train["y"] = pd.to_numeric(df_train["y"], errors='coerce')

# Check if df_train has valid data
if df_train.empty or df_train["y"].isna().all():
    st.error("No valid stock price data available for forecasting. Try another stock.")
    st.stop()

# Train Prophet Model
m = Prophet()
m.fit(df_train)

# Make Future Predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show Forecast Data
st.subheader("📈 Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot Forecast
st.subheader("📉 Stock Price Forecast")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot Forecast Components
st.subheader("📊 Forecast Components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)