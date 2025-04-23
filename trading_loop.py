import yfinance as yf
import time
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
import requests
import pytz

# Telegram config
TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
TELEGRAM_CHAT_ID = '1012793457'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("📨 Telegram message sent")
    else:
        print("❌ Telegram message failed", response.text)

# Settings
TICKERS = [
    "FILATFASH.NS", "SRESTHA.BO", "HARSHILAGR.BO", "GTLINFRA.NS", "ITC.NS",
    "OBEROIRLTY.NS", "JAMNAAUTO.NS", "KSOLVES.NS", "ADANIGREEN.NS",
    "TATAMOTORS.NS", "OLECTRA.NS", "ARE&M.NS", "AFFLE.NS", "BEL.NS",
    "SUNPHARMA.NS", "LAURUSLABS.NS", "RELIANCE.NS", "KRBL.NS", "ONGC.NS",
    "IDFCFIRSTB.NS", "BANKBARODA.NS", "GSFC.NS", "TCS.NS", "INFY.NS"
]
SHARES_TO_BUY = 2
CHECK_INTERVAL = 60 * 5
ATR_MULTIPLIER = 1.5
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Memory
stock_data = {
    ticker: {
        "entry_price": None,
        "holdings": 0,
        "sell_threshold": None,
        "highest_price": None
    }
    for ticker in TICKERS
}
last_actions = {ticker: None for ticker in TICKERS}

# Technical calculations
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
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_volume_spike(data, multiplier=1.5):
    return data['Volume'].iloc[-1] > multiplier * data['Volume'].rolling(20).mean().iloc[-1]

def check_breakout(data, window=5):
    return data['Close'].iloc[-1] > data['High'].rolling(window).max().iloc[-2]

def get_stock_price(ticker):
    data = yf.Ticker(ticker).history(period="1d", interval="1m")
    return data.iloc[-1]['Close'] if not data.empty else None

def get_next_earnings_date(ticker):
    try:
        return yf.Ticker(ticker).calendar.loc['Earnings Date'][0]
    except:
        return None

def is_upcoming_earnings(ticker):
    date = get_next_earnings_date(ticker)
    return date and datetime.now().date() <= date.date() <= (datetime.now() + timedelta(days=2)).date()

def is_friday_exit_time():
    now = datetime.now()
    return now.weekday() == 4 and now.hour == 15 and now.minute >= 20

def close_all_positions():
    for ticker, stock in stock_data.items():
        if stock["holdings"] > 0:
            price = get_stock_price(ticker)
            if price:
                change = ((price - stock["entry_price"]) / stock["entry_price"]) * 100
                msg = f"📤 {ticker}: Weekend exit - Sold {stock['holdings']} @ {price:.2f}\nEntry: {stock['entry_price']:.2f}, Change: {change:.2f}%"
                print(msg)
                send_telegram_message(msg)
            stock.update({"holdings": 0, "entry_price": None, "sell_threshold": None, "highest_price": None})

def get_ist_now():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def main():
    state = {
        "last_alive_915": None,
        "last_alive_300": None,
        "last_print_915": None,
        "last_print_315": None
    }

    while True:
        now_ist = get_ist_now()
        current_time = now_ist.strftime("%H:%M")
        today = now_ist.date()

        # Alive checks
        if current_time == "09:15" and state["last_alive_915"] != today:
            send_telegram_message("✅ Bot is alive – 9:15 AM check-in")
            state["last_alive_915"] = today

        if current_time == "15:00" and state["last_alive_300"] != today:
            send_telegram_message("✅ Bot is alive – 3:00 PM check-in")
            state["last_alive_300"] = today

        # Exit on Friday afternoon
        if is_friday_exit_time():
            print("📆 Friday 3:20 PM – Closing all positions")
            close_all_positions()

        action_changed = False
        table_data = []

        for ticker in TICKERS:
            if is_upcoming_earnings(ticker):
                print(f"🚫 {ticker}: Skipped due to earnings")
                continue

            data = get_historical_data(ticker)
            if data.empty:
                print(f"⚠️ {ticker}: No data")
                continue

            sma_20 = calculate_sma(data, 20).iloc[-1]
            sma_50 = calculate_sma(data, 50).iloc[-1]
            atr = calculate_atr(data, 14).iloc[-1]
            rsi = calculate_rsi(data, 14).iloc[-1]
            price = get_stock_price(ticker)

            if not price:
                print(f"⚠️ {ticker}: No price")
                continue

            stock = stock_data[ticker]
            change = None

            # Buy logic
            if stock["holdings"] == 0 and sma_20 > sma_50 and rsi < RSI_OVERBOUGHT:
                if rsi > RSI_OVERSOLD:
                    stock.update({
                        "entry_price": price,
                        "holdings": SHARES_TO_BUY,
                        "sell_threshold": price - (ATR_MULTIPLIER * atr),
                        "highest_price": price
                    })
                    msg = f"🟢 {ticker} - {price:.2f}, Bought {SHARES_TO_BUY} shares"
                    print(msg)
                    send_telegram_message(msg)

            # Update trailing stop
            if stock["holdings"] > 0:
                if price > stock["highest_price"]:
                    stock["highest_price"] = price
                stock["sell_threshold"] = max(
                    stock["sell_threshold"],
                    stock["highest_price"] - ATR_MULTIPLIER * atr
                )
                change = ((price - stock["entry_price"]) / stock["entry_price"]) * 100

            # Sell logic
            if stock["holdings"] > 0 and price <= stock["sell_threshold"]:
                msg = (
                    f"🔴 {ticker}: Sold {stock['holdings']} shares @ {price:.2f}\n"
                    f"Entry: {stock['entry_price']:.2f}, Change: {change:.2f}%\n"
                    f"Stop loss: {stock['sell_threshold']:.2f}"
                )
                print(msg)
                send_telegram_message(msg)
                stock.update({"holdings": 0, "entry_price": None, "sell_threshold": None, "highest_price": None})

            action = "HOLD" if stock["holdings"] > 0 else "WAIT"
            if last_actions[ticker] != action:
                action_changed = True
                last_actions[ticker] = action

            table_data.append([
                ticker, f"{price:.2f}",
                f"{stock['entry_price']:.2f}" if stock['entry_price'] else "N/A",
                f"{sma_20:.2f}", f"{sma_50:.2f}", f"{atr:.2f}", f"{rsi:.2f}",
                f"{stock['sell_threshold']:.2f}" if stock['sell_threshold'] else "N/A",
                f"{change:.2f}%" if change else "N/A", action
            ])

        should_print_915 = current_time == "09:15" and state["last_print_915"] != today
        should_print_315 = current_time == "15:15" and state["last_print_315"] != today

        if action_changed or should_print_915 or should_print_315:
            print(f"\n📊 {now_ist.strftime('%Y-%m-%d %H:%M:%S')} — Stock Status")
            print(tabulate(table_data, headers=[
                "Ticker", "Current Price", "Entry Price", "20-SMA", "50-SMA",
                "ATR", "RSI", "Sell Threshold", "Change %", "Action"
            ], tablefmt="grid"))

            if should_print_915:
                state["last_print_915"] = today
            if should_print_315:
                state["last_print_315"] = today

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
