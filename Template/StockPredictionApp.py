import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import time
import os
import json

def load_css(file_name="style.css"):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSS file not found: {file_name}")

load_css()  # Load the CSS file

# Constants
START = "2010-01-01"
TODAY = pd.to_datetime("today").strftime('%Y-%m-%d')

# File to store user data
USER_DB_FILE = "users.json"

# Load users from JSON file
def load_users():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as file:
            return json.load(file)
    return {}

# Save users to JSON file
def save_users(users):
    with open(USER_DB_FILE, "w") as file:
        json.dump(users, file)

# Load users at the start
USER_CREDENTIALS = load_users()

# Initialize session state variables
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "remember_me" not in st.session_state:
    st.session_state.remember_me = False

# Welcome Screen
if st.session_state.show_welcome:
    st.image(r"C:\Users\vaishnavi\Downloads\WhatsApp Image 2025-02-26 at 12.22.01.jpeg", width=200)  # Replace 'logo.png' with your actual logo file
    st.title("📈 Stock Sage")
    st.subheader("Your go-to stock and crypto forecasting tool.")
    if st.button("➡️ Continue"):
        st.session_state.show_welcome = False
        st.rerun()
    st.stop()

# Function to handle login
def login():
    st.subheader("🔐 Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    remember_me = st.checkbox("🔄 Remember Me", key="login_remember")
    
    if st.button("🔓 Login"):
        users = load_users()
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.remember_me = remember_me
            
            if remember_me:
                st.session_state.saved_username = username
                st.session_state.saved_password = password

            st.success(f"✅ Welcome, {username}!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Incorrect username or password. Please try again.")

# Function to handle account creation
def create_account():
    st.subheader("🆕 Create Account")
    new_username = st.text_input("👤 Choose a Username")
    new_password = st.text_input("🔑 Choose a Password", type="password")
    
    if st.button("✅ Create Account"):
        users = load_users()
        
        if new_username in users:
            st.error("❌ Username already exists. Please log in.")
        elif new_username and new_password:
            users[new_username] = new_password
            save_users(users)
            st.success("🎉 Account created successfully! Please log in.")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Please enter both a username and password.")

# Check for remembered login details
if "saved_username" in st.session_state and "saved_password" in st.session_state:
    users = load_users()
    if st.session_state.saved_username in users and users[st.session_state.saved_username] == st.session_state.saved_password:
        st.session_state.logged_in = True
        st.session_state.username = st.session_state.saved_username

# Show login and create account on the same page
if not st.session_state.logged_in:
    login()
    st.markdown("---")
    st.write("Can't log in? Create an account below.")
    create_account()
    st.stop()

# Function to handle logout
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.remember_me = False
    st.session_state.pop("saved_username", None)
    st.session_state.pop("saved_password", None)
    st.session_state["logout_trigger"] = True  # Set a flag for rerun

# Show Logout Button
st.sidebar.button("🚪 Logout", on_click=logout)

# Check if logout was triggered
if st.session_state.get("logout_trigger"):
    del st.session_state["logout_trigger"]  # Remove flag after rerun
    st.rerun()

# Streamlit App Title (Only visible after login)
st.title(f"🚀 Welcome, {st.session_state.username}!")

# HTML & JavaScript Integration
import streamlit as st
import time

# Streamlit App Title
st.markdown("""
    <h2 style='text-align: center;'>📊 Smart Stock Dashboard</h2>
""", unsafe_allow_html=True)

# HTML & JavaScript Integration
html_code = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        #typewriter-text { font-size: 22px; font-weight: bold; color: #3498db; }
        #clock { font-size: 24px; font-weight: bold; margin-top: 10px; }
        .stock-ticker { white-space: nowrap; overflow: hidden; }
        .sentiment { font-size: 20px; font-weight: bold; }
        .bullish { color: green; }
        .bearish { color: red; }
        .neutral { color: orange; }
    </style>
</head>
<body>
    <h3 id="typewriter-text"></h3>
    <p id="clock"></p>
    <div class="stock-ticker" id="ticker">📈 Loading stock data...</div>
    <p class="sentiment" id="sentiment">📊 Market Sentiment: Neutral</p>

    <script>
        // Typewriter Effect (Repeats)
        const texts = [
            "📈 The market waits for no one... Stay ahead with insights! 💹",
            "💰 Smart decisions today shape your financial future! 🔥"
        ];
        let textIndex = 0;
        let charIndex = 0;
        function typeWriter() {
            let currentText = texts[textIndex];
            document.getElementById("typewriter-text").innerHTML = currentText.substring(0, charIndex++);
            if (charIndex > currentText.length) {
                charIndex = 0;
                textIndex = (textIndex + 1) % texts.length;
                setTimeout(typeWriter, 1000);
            } else {
                setTimeout(typeWriter, 150);
            }
        }
        typeWriter();

        // Live Digital Clock
        function updateClock() {
            const now = new Date();
            document.getElementById("clock").innerText = now.toLocaleTimeString();
        }
        setInterval(updateClock, 1000);
        updateClock();

        // Stock Market Ticker (Mock Data)
        const stockPrices = ["AAPL: $150.23", "GOOGL: $2750.65", "TSLA: $849.99", "AMZN: $3450.12", "MSFT: $299.99"];
        let tickerIndex = 0;
        function updateTicker() {
            document.getElementById("ticker").innerText = stockPrices[tickerIndex];
            tickerIndex = (tickerIndex + 1) % stockPrices.length;
        }
        setInterval(updateTicker, 2000);

        // Stock Sentiment Animation
        const sentiments = ["📈 Bullish", "📉 Bearish", "📊 Neutral"];
        const sentimentColors = ["bullish", "bearish", "neutral"];
        let sentimentIndex = 0;
        function updateSentiment() {
            let sentimentElement = document.getElementById("sentiment");
            sentimentElement.innerText = "📊 Market Sentiment: " + sentiments[sentimentIndex];
            sentimentElement.className = "sentiment " + sentimentColors[sentimentIndex];
            sentimentIndex = (sentimentIndex + 1) % sentiments.length;
        }
        setInterval(updateSentiment, 3000);
    </script>
</body>
</html>
"""

# Embed HTML in Streamlit
st.components.v1.html(html_code, height=350)



# Sidebar Navigation
st.sidebar.header(f"👋 Hello, {st.session_state.get('username', 'User')}")
feature_choice = st.sidebar.radio("📌 Select Feature", ["Stock Forecast", "Crypto Tracker", "Portfolio Simulator"])

# List of stocks & cryptocurrencies
stocks = ["AAPL", "GOOG", "TSLA", "AMZN", "NVDA", "META", "MSFT"]
cryptos = ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD"]
selected_asset = st.sidebar.selectbox("📊 Choose Asset", stocks + cryptos)

# Function to load stock/crypto data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    
    # Handle MultiIndex issue
    if isinstance(data.columns, pd.MultiIndex):
        data = data[[('Date', ''), ('Close', ticker)]].copy()
        data.columns = ['Date', 'Close']
    
    return data

# Load data
data_load_state = st.text("Loading data...")
data = load_data(selected_asset)
data_load_state.text("Loading data... done!")

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Check if 'Close' column exists
if 'Close' in data.columns:
    data.dropna(subset=['Close'], inplace=True)
else:
    st.error("Column 'Close' is missing in stock data. Cannot proceed.")
    st.stop()

if feature_choice == "Stock Forecast":
    # Show raw data
    st.subheader("📊 Raw Data")
    st.write(data.tail())

    # Function to plot historical prices
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='green')))
        fig.update_layout(title_text="📅 Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
    
    plot_raw_data()

    st.subheader(f"📈 Stock Prediction for {selected_asset}")
    n_years = st.slider("🎮 Years of prediction:", 1, 5)
    period = n_years * 365
    
    # Prepare Data for Forecasting
    df_train = data[['Date', 'Close']].copy()
    df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    df_train.dropna(inplace=True)
    df_train["y"] = pd.to_numeric(df_train["y"], errors='coerce')
    
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

elif feature_choice == "Crypto Tracker":
    st.subheader(f"💰 Real-time Crypto Price for {selected_asset}")
    st.subheader("📊 Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Crypto Price", line=dict(color='blue')))
    fig.update_layout(title_text="📅 Crypto Price Trends", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"📈 Latest Price of {selected_asset}: *${data['Close'].iloc[-1]:,.2f}*")

elif feature_choice == "Portfolio Simulator":
    st.subheader("💰 Virtual Portfolio Simulator")
    investment = st.number_input("💸 Enter Investment Amount ($)", min_value=100, value=1000, step=100)
    starting_price = data['Close'].iloc[0]
    current_price = data['Close'].iloc[-1]
    
    profit_loss = ((current_price - starting_price) / starting_price) * 100
    final_value = investment * (1 + profit_loss / 100)
    
    st.write(f"📈 Initial Investment: *${investment:,.2f}*")
    st.write(f"📊 Current Value: *${final_value:,.2f}*")
    st.write(f"📉 Profit/Loss: *{profit_loss:.2f}%*")

    # Portfolio Growth Graph
    st.subheader("📈 Portfolio Growth Over Time")
    data["Portfolio Value"] = investment * (data["Close"] / starting_price)
    fig = px.line(data, x="Date", y="Portfolio Value", title="Investment Growth Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Investment Guidance
    st.subheader("💡 Investment Guidance")
    if profit_loss > 20:
        st.success("🚀 Your investment is performing **very well**! Consider holding or reinvesting.")
    elif profit_loss > 0:
        st.info("📈 Your investment is **growing**! Stay invested for potential gains.")
    elif profit_loss > -10:
        st.warning("⚠️ Your investment is slightly down. Keep an eye on market trends.")
    else:
        st.error("📉 Your investment has **significant losses**. Consider reviewing your strategy.")
