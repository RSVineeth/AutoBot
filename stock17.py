import yfinance as yf
import time
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import requests

# Twilio Credentials
TWILIO_SID = "AC0fcf79de541f9bb6dad504627f0f7489"
TWILIO_AUTH_TOKEN = "3c6eb41be312a67ba3dd9c20e9de31a1"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"
RECEIVER_WHATSAPP_NUMBER = "whatsapp:+917396730966"

TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
TELEGRAM_CHAT_ID = '1012793457'

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# def send_whatsapp_message(message):
#     client.messages.create(
#         from_=TWILIO_WHATSAPP_NUMBER,
#         body=message,
#         to=RECEIVER_WHATSAPP_NUMBER
#     )



# def send_whatsapp_message(message):
#     try:
#         client.messages.create(
#             from_=TWILIO_WHATSAPP_NUMBER,
#             body=message,
#             to=RECEIVER_WHATSAPP_NUMBER
#         )
#         print("📤 WhatsApp message sent")
#     except TwilioRestException as e:
#         print(f"⚠️ WhatsApp failed: {e.msg}. Sending via Telegram instead.")
#         send_telegram_message(message)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("📨 Telegram message sent")
    else:
        print("❌ Telegram message failed", response.text)

# Parameters
TICKERS = ["FILATFASH.NS", "SRESTHA.BO", "HARSHILAGR.BO", "GTLINFRA.NS", "ITC.NS", "OBEROIRLTY.NS", "JAMNAAUTO.NS", "KSOLVES.NS", "ADANIGREEN.NS", "TATAMOTORS.NS", "OLECTRA.NS", "ARE&M.NS", "AFFLE.NS", "BEL.NS", "SUNPHARMA.NS", "LAURUSLABS.NS", "RELIANCE.NS", "KRBL.NS", "ONGC.NS", "IDFCFIRSTB.NS", "BANKBARODA.NS", "GSFC.NS", "TCS.NS", "INFY.NS"]  # List of stocks
SHARES_TO_BUY = 2
CHECK_INTERVAL = 60 * 5
ATR_MULTIPLIER = 1.5
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

stock_data = {
    ticker: {"entry_price": None, "holdings": 0, "sell_threshold": None, "highest_price": None}
    for ticker in TICKERS
}

def get_historical_data(ticker, period="3mo"):
    stock_info = yf.Ticker(ticker)
    history = stock_info.history(period=period)
    if hasattr(history.index, 'tz'):
        history.index = history.index.tz_localize(None)
    return history

def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window=20):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_atr(data, window=14):
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift())
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift())
    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    return data['True Range'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_volume_spike(data, multiplier=1.5):
    recent_volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
    return recent_volume > (multiplier * avg_volume)

def check_breakout(data, window=5):
    recent_high = data['High'].rolling(window=window).max().iloc[-2]
    current_close = data['Close'].iloc[-1]
    return current_close > recent_high

def get_stock_price(ticker):
    data = yf.Ticker(ticker).history(period="1d", interval="1m")
    if not data.empty:
        return data.iloc[-1]['Close']
    return None

def get_next_earnings_date(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings_date = stock.calendar.loc['Earnings Date'][0]
        return earnings_date
    except:
        return None

def is_upcoming_earnings(ticker):
    earnings_date = get_next_earnings_date(ticker)
    if pd.isna(earnings_date):
        return False
    return datetime.now().date() <= earnings_date.date() <= (datetime.now() + timedelta(days=2)).date()

def is_friday_exit_time():
    now = datetime.now()
    return now.weekday() == 4 and now.hour == 15 and now.minute >= 20

def close_all_positions():
    for ticker, stock in stock_data.items():
        if stock["holdings"] > 0:
            current_price = get_stock_price(ticker)
            if current_price:
                change_percent = ((current_price - stock["entry_price"]) / stock["entry_price"]) * 100
                message = (f"\ud83d\udce4 {ticker}: Weekend exit - Sold {stock['holdings']} shares at {current_price:.2f}\n"
                           f"Entry: {stock['entry_price']:.2f}, Change: {change_percent:.2f}%")
                print(message)
                send_telegram_message(message)

            stock["holdings"] = 0
            stock["entry_price"] = None
            stock["sell_threshold"] = None
            stock["highest_price"] = None

while True:
    if is_friday_exit_time():
        print("\ud83d\uddd3\ufe0f It's Friday 3:20 PM — closing all positions to avoid weekend gap risk.")
        close_all_positions()

    table_data = []
    for ticker in TICKERS:
        if is_upcoming_earnings(ticker):
            print(f"\u26d4\ufe0f {ticker}: Earnings event approaching. Skipping entry.")
            continue

        data = get_historical_data(ticker)
        if data.empty:
            print(f"⚠️ {ticker}: No historical data available")
            continue

        sma_20 = calculate_sma(data, 20).iloc[-1]
        sma_50 = calculate_sma(data, 50).iloc[-1]
        ema_20 = calculate_ema(data, 20).iloc[-1]
        ema_50 = calculate_ema(data, 50).iloc[-1]
        atr = calculate_atr(data, 14).iloc[-1]
        rsi = calculate_rsi(data, 14).iloc[-1]
        volume_spike = check_volume_spike(data)
        breakout = check_breakout(data)

        price = get_stock_price(ticker)
        if price is None:
            print(f"⚠️ {ticker}: No price data available")
            continue

        stock = stock_data[ticker]
        change_percent = None

        # Buy condition: SMA confirmation, RSI check & no holdings
        if stock["holdings"] == 0 and sma_20 > sma_50 and rsi < RSI_OVERBOUGHT:
            if rsi < RSI_OVERSOLD:
                print(f"⚠️ {ticker}: RSI is too low (<{RSI_OVERSOLD}), waiting for confirmation")
            else:
                stock["entry_price"] = price
                stock["holdings"] += SHARES_TO_BUY
                stock["sell_threshold"] = price - (ATR_MULTIPLIER * atr)  # ATR Stop Loss
                stock["highest_price"] = price
                
                # Send Buy message
                buy_message = f"{ticker} - {price:.2f}, Bought {SHARES_TO_BUY} shares"
                print(f"🟢 {buy_message}")
                send_telegram_message(buy_message)


        if stock["holdings"] > 0:
            if price > stock["highest_price"]:
                stock["highest_price"] = price
            stock["sell_threshold"] = max(stock["sell_threshold"], stock["highest_price"] - (ATR_MULTIPLIER * atr))
            if stock["entry_price"]:
                change_percent = ((price - stock["entry_price"]) / stock["entry_price"]) * 100

        if stock["holdings"] > 0 and price <= stock["sell_threshold"]:
            sell_message = (f"🔴 {ticker}: Sold {stock['holdings']} shares at {price:.2f}\n"
                            f"Entry Price: {stock['entry_price']:.2f}, Change: {change_percent:.2f}%\n"
                            f"Stop Loss Triggered at: {stock['sell_threshold']:.2f}")
            print(sell_message)
            send_telegram_message(sell_message)
            stock["holdings"] = 0
            stock["entry_price"] = None
            stock["sell_threshold"] = None
            stock["highest_price"] = None

        action = "HOLD" if stock["holdings"] > 0 else "WAIT FOR ENTRY"
        table_data.append([
            ticker,
            f"{price:.2f}",
            f"{stock['entry_price']:.2f}" if stock['entry_price'] else "N/A",
            f"{sma_20:.2f}",
            f"{sma_50:.2f}",
            f"{atr:.2f}",
            f"{rsi:.2f}",
            f"{stock['sell_threshold']:.2f}" if stock['sell_threshold'] else "N/A",
            f"{change_percent:.2f}%" if change_percent is not None else "N/A",
            action
        ])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nTimestamp: {timestamp}")

    headers = ["Ticker", "Current Price", "Entry Price", "20-SMA", "50-SMA", "ATR", "RSI", "Sell Threshold", "Change %", "Action"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    time.sleep(CHECK_INTERVAL)