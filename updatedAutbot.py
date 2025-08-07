import yfinance as yf
import time
import pandas as pd
from ta.trend import ADXIndicator
from datetime import datetime, timedelta
from tabulate import tabulate
import requests
import pytz
import os
import threading
import logging

# Setup logging to stdout with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("yfinance").disabled = True

# Optional: Replace print with logging if preferred
print = logging.info

# Telegram config
TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
TELEGRAM_CHAT_ID = ['1012793457','1209666577']

def send_telegram_message(message):
    for chat_id in TELEGRAM_CHAT_ID:
        chat_id = chat_id.strip()
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
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
VOLUME_SPIKE_MULTIPLIER = 1.5
BREAKOUT_WINDOW = 5

# Memory
stock_data = {
    ticker: {
        "entry_price": None,
        "holdings": 0,
        "sell_threshold": None,
        "highest_price": None,
        "notified_52w_high": False
    }
    for ticker in TICKERS
}
last_actions = {ticker: None for ticker in TICKERS}

def telegram_listener():
    print("👂 Telegram listener started")
    last_update_id = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            if last_update_id:
                url += f"?offset={last_update_id + 1}"

            response = requests.get(url).json()

            if "result" in response:
                for update in response["result"]:
                    last_update_id = update["update_id"]
                    if "message" in update and "text" in update["message"]:
                        ticker_input = update["message"]["text"].strip().upper()
                        if ticker_input in TICKERS:
                            print_ticker_table(ticker_input)
                        else:
                            send_telegram_message(f"❌ Unknown ticker: {ticker_input}")

        except Exception as e:
            print(f"⚠️ Telegram listener error: {e}")
        time.sleep(5)

def print_ticker_table(ticker):
    stock = stock_data.get(ticker)
    if not stock:
        print(f"⚠️ {ticker} not in stock_data")
        return

    data = get_historical_data(ticker)
    if data is None or data.empty:
        print(f"⚠️ No data for {ticker}")
        return

    price = get_stock_price(ticker)
    if not price:
        print(f"⚠️ No price for {ticker}")
        return

    adx = calculate_adx(data).iloc[-1]
    ema_20 = calculate_ema(data, 20).iloc[-1]
    sma_20 = calculate_sma(data, 20).iloc[-1]
    sma_50 = calculate_sma(data, 50).iloc[-1]
    atr = calculate_atr(data, 14).iloc[-1]
    rsi = calculate_rsi(data, 14).iloc[-1]
    macd_line, signal_line, macd_hist = calculate_macd(data)
    macd_current = macd_line.iloc[-1]
    signal_current = signal_line.iloc[-1]
    volume_spike = check_volume_spike(data, VOLUME_SPIKE_MULTIPLIER)
    breakout = check_breakout(data, BREAKOUT_WINDOW)
    change = ((price - stock["entry_price"]) / stock["entry_price"]) * 100 if stock["entry_price"] else None

    table_data = [[
        ticker, f"{price:.2f}",
        f"{stock['entry_price']:.2f}" if stock['entry_price'] else "N/A",
        f"{ema_20:.2f}", f"{sma_20:.2f}", f"{sma_50:.2f}", f"{macd_current:.2f}", f"{signal_current:.2f}",
        f"{atr:.2f}", f"{rsi:.2f}", f"{adx:.2f}",
        "✅" if volume_spike else "❌", "✅" if breakout else "❌",
        f"{stock['sell_threshold']:.2f}" if stock['sell_threshold'] else "N/A",
        f"{change:.2f}%" if change else "N/A",
        "HOLD" if stock['holdings'] > 0 else "WAIT"
    ]]

    print(tabulate(table_data, headers=[
        "Ticker", "Current Price", "Entry Price", "20-EMA", "20-SMA", "50-SMA", "MACD", "Signal",
        "ATR", "RSI", "ADX", "Vol Spike", "Breakout", "Sell Threshold", "Change %", "Action"
    ], tablefmt="grid"))

# Technical calculations
def get_historical_data(ticker, period="3mo"):
    try:
        stock_info = yf.Ticker(ticker)
        history = stock_info.history(period=period)
        if history.empty:
            print(f"⚠️ {ticker}: No price data found (period={period}) – removing.")
            return None
        if hasattr(history.index, 'tz'):
            history.index = history.index.tz_localize(None)
        return history
    except Exception as e:
        print(f"❌ {ticker}: Error fetching data: {e}")
        return None
    
def get_annual_high(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return None
        return hist["High"].max()
    except Exception as e:
        print(f"❌ {ticker}: Error fetching 52-week high: {e}")
        return None

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
    rs = gain / loss.replace(0, 1e-10)  # Prevent division by zero
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_adx(data, window=14):
    adx = ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=window)
    return adx.adx()

# NEW: Volume and Breakout Analysis Functions
def check_volume_spike(data, multiplier=1.5):
    """Check if current volume is significantly higher than average"""
    if len(data) < 20:
        return False
    current_volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    return current_volume > multiplier * avg_volume

def check_breakout(data, window=5):
    """Check if current price breaks above recent high"""
    if len(data) < window + 1:
        return False
    recent_high = data['High'].rolling(window).max().iloc[-2]
    current_close = data['Close'].iloc[-1]
    return current_close > recent_high

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
            stock.update({"holdings": 0, "entry_price": None, "sell_threshold": None, "highest_price": None, "notified_52w_high": False})

def get_ist_now():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def main():
    print("✅ Enhanced trading_loop.py: main() started")
    print(f"📊 Loaded TICKERS: {TICKERS}")

    state = {
        "last_alive_915": None,
        "last_alive_300": None,
        "last_print_915": None,
        "last_print_315": None
    }

    while True:
        threading.Thread(target=telegram_listener, daemon=True).start()
        now_ist = get_ist_now()
        current_time = now_ist.strftime("%H:%M")
        today = now_ist.date()

        # Alive checks (within time ranges, once per day)
        if now_ist.hour == 9 and 15 <= now_ist.minute <= 30:
            if state["last_alive_915"] != today:
                send_telegram_message("✅ Bot is alive – morning check")
                print("✅ Bot is alive – morning check")
                state["last_alive_915"] = today

        if now_ist.hour == 15 and 0 <= now_ist.minute <= 15:
            if state["last_alive_300"] != today:
                send_telegram_message("✅ Bot is alive – afternoon check")
                print("✅ Bot is alive – afternoon check")
                state["last_alive_300"] = today

        # Friday exit logic
        if is_friday_exit_time():
            print("📆 Friday 3:20 PM – Closing all positions")
            close_all_positions()

        action_changed = False
        table_data = []
        valid_tickers = []

        for ticker in TICKERS:
            if is_upcoming_earnings(ticker):
                print(f"🚫 {ticker}: Skipped due to earnings")
                continue

            data = get_historical_data(ticker)
            if data is None or data.empty:
                print(f"⚠️ {ticker}: No data — removing from tracking")
                continue

            price = get_stock_price(ticker)
            if not price:
                print(f"⚠️ {ticker}: No price")
                continue

            if ticker not in stock_data:
                print(f"❌ {ticker}: Missing in stock_data — skipping")
                continue

            valid_tickers.append((ticker, data, price))

        for ticker, data, price in valid_tickers:
            adx = calculate_adx(data).iloc[-1]
            ema_20 = calculate_ema(data, 20).iloc[-1]
            sma_20 = calculate_sma(data, 20).iloc[-1]
            sma_50 = calculate_sma(data, 50).iloc[-1]
            atr = calculate_atr(data, 14).iloc[-1]
            rsi = calculate_rsi(data, 14).iloc[-1]
            
            # NEW: Volume and breakout analysis
            volume_spike = check_volume_spike(data, VOLUME_SPIKE_MULTIPLIER)
            breakout = check_breakout(data, BREAKOUT_WINDOW)

            stock = stock_data[ticker]
            change = None

            macd_line, signal_line, macd_hist = calculate_macd(data)
            macd_current = macd_line.iloc[-1]
            signal_current = signal_line.iloc[-1]

            # ENHANCED Buy logic with volume and breakout confirmation
            if stock["holdings"] == 0 and ema_20 > sma_50 and rsi < RSI_OVERBOUGHT and adx > 20:
                if rsi > RSI_OVERSOLD and macd_current > signal_current:
                    # NEW: Additional confirmation from volume spike OR breakout
                    if volume_spike or breakout:
                        stock.update({
                            "entry_price": price,
                            "holdings": SHARES_TO_BUY,
                            "sell_threshold": price - (ATR_MULTIPLIER * atr),
                            "highest_price": price
                        })
                        
                        # Enhanced buy message with confirmation signals
                        confirmations = []
                        if volume_spike:
                            confirmations.append("Volume Spike")
                        if breakout:
                            confirmations.append("Breakout")
                        
                        msg = (f"🟢 {ticker} - {price:.2f}\n"
                               f"ATR: {atr:.2f}, Sell Threshold: {price - (ATR_MULTIPLIER * atr):.2f}\n"
                               f"Confirmations: {', '.join(confirmations)}")
                        print(msg)
                        send_telegram_message(msg)
                    else:
                        print(f"⚠️ {ticker}: All conditions met but waiting for volume spike or breakout confirmation")

            # Update trailing stop
            if stock["holdings"] > 0:
                if price > stock["highest_price"]:
                    stock["highest_price"] = price
                stock["sell_threshold"] = max(
                    stock["sell_threshold"],
                    stock["highest_price"] - ATR_MULTIPLIER * atr
                )
                change = ((price - stock["entry_price"]) / stock["entry_price"]) * 100

            # Check for annual high
            annual_high = get_annual_high(ticker)
            if stock["holdings"] > 0 and annual_high and abs(price - annual_high) < 0.5:
                if not stock.get("notified_52w_high", False):
                    msg = (
                        f"📈 {ticker} has reached its 52-week high at {price:.2f}.\n"
                        f"Entry: {stock['entry_price']:.2f}, Change: {change:.2f}%\n"
                        f"Volume Spike: {'✅' if volume_spike else '❌'}\n"
                        f"Would you like to SELL or HOLD?"
                    )
                    print(msg)
                    send_telegram_message(msg)
                    stock["notified_52w_high"] = True

            # Sell logic
            if stock["holdings"] > 0 and price <= stock["sell_threshold"]:
                msg = (
                    f"🔴 {ticker}: Sold shares at {price:.2f}\n"
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
                f"{ema_20:.2f}", f"{sma_20:.2f}", f"{sma_50:.2f}", f"{atr:.2f}", f"{rsi:.2f}",
                f"{macd_current:.2f}", f"{signal_current:.2f}", f"{adx:.2f}",
                "✅" if volume_spike else "❌", "✅" if breakout else "❌",
                f"{stock['sell_threshold']:.2f}" if stock['sell_threshold'] else "N/A",
                f"{change:.2f}%" if change else "N/A", action
            ])

        should_print_915 = current_time == "09:15" and state["last_print_915"] != today
        should_print_315 = current_time == "15:15" and state["last_print_315"] != today

        if action_changed or should_print_915 or should_print_315:
            print(tabulate(table_data, headers=[
                "Ticker", "Current Price", "Entry Price", "20-EMA", "20-SMA", "50-SMA", "ATR", "RSI",
                "MACD", "Signal", "ADX", "Vol Spike", "Breakout", "Sell Threshold", "Change %", "Action"
            ], tablefmt="grid"))

            if should_print_915:
                state["last_print_915"] = today
            if should_print_315:
                state["last_print_315"] = today

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()