import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import warnings
from datetime import datetime, timedelta
import talib
import schedule
import threading
from typing import Dict, List, Optional
from tabulate import tabulate
import atexit
import signal
import sys
import logging

warnings.filterwarnings('ignore')

# ============================
# LOGGING CONFIGURATION
# ============================

# Setup logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler('trading_bot.log', mode='a')  # File output
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

# Suppress noisy external library logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("yfinance").disabled = True

# ============================
# CONFIGURATION
# ============================

# Telegram Configuration
TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
TELEGRAM_CHAT_ID = ['1012793457','1209666577']

# Trading Configuration
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS', 
           'BHARTIARTL.NS', 'SBIN.NS', 'LT.NS', 'HCLTECH.NS', 'WIPRO.NS']

CHECK_INTERVAL = 60 * 5  # 5 minutes
SHARES_TO_BUY = 2
ATR_MULTIPLIER = 1.5
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Market Hours (IST)
MARKET_START = "00:15"
MARKET_END = "23:45"
ALIVE_CHECK_MORNING = "09:15"
ALIVE_CHECK_EVENING = "15:00"

# ============================
# GLOBAL VARIABLES
# ============================

class StockMemory:
    def __init__(self):
        self.holdings = {}  # {ticker: {'shares': int, 'entry_price': float}}
        self.sell_thresholds = {}  # {ticker: float}
        self.highest_prices = {}  # {ticker: float}
        self.alerts_sent = {}  # {ticker: {'52w_high': bool}}
        self.last_action_status = {}  # {ticker: 'HOLD'/'WAIT'}
        self.last_alive_check = None
        self.session_start_time = datetime.now()
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0

memory = StockMemory()

# ============================
# EXIT HANDLERS
# ============================

def cleanup_and_exit():
    """Clean exit with summary"""
    logger.info("Bot shutting down...")
    print_final_summary()
    send_telegram_message("🛑 *Bot Stopped*\nTrading session ended")
    sys.exit(0)

def setup_exit_handlers():
    """Setup graceful exit handlers"""
    def signal_handler(sig, frame):
        cleanup_and_exit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(print_final_summary)

def print_final_summary():
    """Print final session summary"""
    try:
        session_duration = datetime.now() - memory.session_start_time
        active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
        
        summary_lines = [
            "="*80,
            "FINAL SESSION SUMMARY",
            "="*80,
            f"Session Duration: {session_duration}",
            f"Total Trades: {memory.total_trades}",
            f"Profitable Trades: {memory.profitable_trades}",
            f"Win Rate: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "Win Rate: 0%",
            f"Total P&L: {memory.total_pnl:.2f}",
            f"Active Positions: {active_positions}",
            "="*80
        ]
        
        for line in summary_lines:
            logger.info(line)
            
    except Exception as e:
        logger.error(f"Error in final summary: {e}")

# ============================
# TELEGRAM FUNCTIONS
# ============================

def send_telegram_message(message: str):
    """Send message to all configured Telegram chats"""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logger.info(f"[TELEGRAM] {message}")
        return
    
    for chat_id in TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to send telegram message: {response.text}")
            else:
                logger.debug(f"Telegram message sent to {chat_id}")
        except Exception as e:
            logger.error(f"Telegram error for chat {chat_id}: {e}")

def send_alive_notification():
    """Send bot alive notification"""
    current_time = datetime.now().strftime("%H:%M")
    active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
    
    message = f"✅ *Stock Trading Bot is ALIVE* - {current_time}\n"
    message += f"📊 Monitoring {len(TICKERS)} stocks\n"
    message += f"💼 Active positions: {active_positions}\n"
    message += f"💰 Session P&L: {memory.total_pnl:.2f}"
    
    send_telegram_message(message)
    memory.last_alive_check = datetime.now()
    logger.info(f"Alive notification sent at {current_time}")

# ============================
# TECHNICAL INDICATORS
# ============================

def calculate_indicators(df: pd.DataFrame) -> Dict:
    """Calculate technical indicators"""
    try:
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        volume = df['Volume'].values
        
        if len(close_prices) < 50:
            logger.warning("Not enough data for indicators calculation")
            return {}
        
        # Simple Moving Averages
        sma_20 = talib.SMA(close_prices, timeperiod=20)
        sma_50 = talib.SMA(close_prices, timeperiod=50)
        
        # RSI
        rsi = talib.RSI(close_prices, timeperiod=14)
        
        # ATR
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Volume analysis
        volume_sma = talib.SMA(volume.astype(float), timeperiod=20)
        volume_spike = False
        if len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) and volume_sma[-1] > 0:
            volume_spike = volume[-1] > (volume_sma[-1] * 1.5)
        
        def safe_extract(arr, default=None):
            if arr is None or len(arr) == 0:
                return default
            val = arr[-1]
            return float(val) if not np.isnan(val) else default
        
        result = {
            'sma_20': safe_extract(sma_20),
            'sma_50': safe_extract(sma_50),
            'rsi': safe_extract(rsi),
            'atr': safe_extract(atr),
            'volume_spike': volume_spike,
            '52w_high': float(df['High'].max()),
            '52w_low': float(df['Low'].min())
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

# ============================
# DATA FETCHING
# ============================

def get_stock_data(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def get_realtime_data(ticker: str) -> Optional[Dict]:
    """Get real-time stock data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1m")
        if df.empty:
            return None
        
        current_price = df['Close'].iloc[-1]
        return {
            'price': current_price,
            'volume': df['Volume'].iloc[-1],
            'high': df['High'].iloc[-1],
            'low': df['Low'].iloc[-1]
        }
    except Exception as e:
        logger.error(f"Error getting real-time data for {ticker}: {e}")
        return None

def has_earnings_soon(ticker: str) -> bool:
    """Check if stock has earnings in next 2 days"""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is not None and not calendar.empty:
            next_earnings = pd.to_datetime(calendar.index[0])
            days_until = (next_earnings - datetime.now()).days
            if days_until <= 2:
                logger.info(f"{ticker} has earnings in {days_until} days")
                return True
    except Exception as e:
        logger.debug(f"No earnings data for {ticker}: {e}")
    return False

# ============================
# TRADING LOGIC
# ============================

def should_buy(ticker: str, indicators: Dict, current_price: float) -> bool:
    """Determine if we should buy the stock"""
    try:
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            return False
        
        if has_earnings_soon(ticker):
            logger.info(f"{ticker}: Skipping due to upcoming earnings")
            return False
        
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        rsi = indicators.get('rsi')
        
        if None in [sma_20, sma_50, rsi]:
            logger.warning(f"{ticker}: Missing indicators for buy decision")
            return False
        
        trend_bullish = sma_20 > sma_50
        rsi_good = RSI_OVERSOLD < rsi < RSI_OVERBOUGHT
        
        logger.debug(f"{ticker}: SMA20={sma_20:.2f}, SMA50={sma_50:.2f}, RSI={rsi:.1f}, Trend={trend_bullish}, RSI_OK={rsi_good}")
        
        return trend_bullish and rsi_good
        
    except Exception as e:
        logger.error(f"Error in should_buy for {ticker}: {e}")
        return False

def should_sell(ticker: str, current_price: float) -> bool:
    """Determine if we should sell the stock"""
    if ticker not in memory.holdings or memory.holdings[ticker].get('shares', 0) == 0:
        return False
    
    if ticker in memory.sell_thresholds:
        should_sell_flag = current_price <= memory.sell_thresholds[ticker]
        if should_sell_flag:
            logger.info(f"{ticker}: Price {current_price:.2f} hit stop-loss {memory.sell_thresholds[ticker]:.2f}")
        return should_sell_flag
    
    return False

def execute_buy(ticker: str, current_price: float, indicators: Dict):
    """Execute buy order"""
    atr = indicators.get('atr', 0)
    
    if atr is None or atr <= 0:
        atr = current_price * 0.02
    
    memory.holdings[ticker] = {
        'shares': SHARES_TO_BUY,
        'entry_price': current_price
    }
    
    memory.sell_thresholds[ticker] = current_price - (ATR_MULTIPLIER * atr)
    memory.highest_prices[ticker] = current_price
    
    if ticker not in memory.alerts_sent:
        memory.alerts_sent[ticker] = {'52w_high': False}
    
    memory.total_trades += 1
    
    symbol = ticker.replace('.NS', '').replace('.BO', '')
    rsi_val = indicators.get('rsi', 0)
    rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
    
    message = f"🟢 *BUY SIGNAL*\n"
    message += f"📈 {symbol} - {current_price:.2f}\n"
    message += f"💰 Bought {SHARES_TO_BUY} shares\n"
    message += f"🛑 Stop-loss: {memory.sell_thresholds[ticker]:.2f}\n"
    message += f"📊 RSI: {rsi_str}"
    
    send_telegram_message(message)
    logger.info(f"[BUY] {symbol} @ {current_price:.2f} | Stop-loss: {memory.sell_thresholds[ticker]:.2f}")

def execute_sell(ticker: str, current_price: float, reason: str = "Stop-loss"):
    """Execute sell order"""
    if ticker not in memory.holdings:
        return
    
    shares = memory.holdings[ticker].get('shares', 0)
    entry_price = memory.holdings[ticker].get('entry_price', 0)
    
    if shares == 0:
        return
    
    # Calculate P&L
    total_change = (current_price - entry_price) * shares
    change_percent = ((current_price - entry_price) / entry_price) * 100
    
    # Update session statistics
    memory.total_pnl += total_change
    if total_change > 0:
        memory.profitable_trades += 1
    
    # Clear position
    memory.holdings[ticker] = {'shares': 0, 'entry_price': 0}
    if ticker in memory.sell_thresholds:
        del memory.sell_thresholds[ticker]
    if ticker in memory.highest_prices:
        del memory.highest_prices[ticker]
    
    memory.alerts_sent[ticker] = {'52w_high': False}
    
    symbol = ticker.replace('.NS', '').replace('.BO', '')
    profit_emoji = "💚" if total_change >= 0 else "❌"
    
    message = f"🔴 *SELL SIGNAL* - {reason}\n"
    message += f"📉 {symbol} - {current_price:.2f}\n"
    message += f"💼 Sold {shares} shares\n"
    message += f"{profit_emoji} P&L: {total_change:.2f} ({change_percent:+.2f}%)"
    
    send_telegram_message(message)
    logger.info(f"[SELL] {symbol} @ {current_price:.2f} | P&L: {total_change:.2f} ({change_percent:+.2f}%)")

def update_trailing_stop(ticker: str, current_price: float, atr: float):
    """Update trailing stop-loss"""
    if ticker not in memory.holdings or memory.holdings[ticker].get('shares', 0) == 0:
        return
    
    if atr is None or atr <= 0:
        atr = current_price * 0.02
    
    if ticker not in memory.highest_prices:
        memory.highest_prices[ticker] = current_price
    else:
        old_highest = memory.highest_prices[ticker]
        memory.highest_prices[ticker] = max(memory.highest_prices[ticker], current_price)
        if current_price > old_highest:
            logger.debug(f"{ticker}: New high price {current_price:.2f}")
    
    new_stop = memory.highest_prices[ticker] - (ATR_MULTIPLIER * atr)
    
    if ticker not in memory.sell_thresholds:
        memory.sell_thresholds[ticker] = new_stop
    else:
        old_stop = memory.sell_thresholds[ticker]
        memory.sell_thresholds[ticker] = max(memory.sell_thresholds[ticker], new_stop)
        if new_stop > old_stop:
            logger.debug(f"{ticker}: Trailing stop updated from {old_stop:.2f} to {new_stop:.2f}")

def check_52w_high_alert(ticker: str, current_price: float, indicators: Dict):
    """Check and send 52-week high alert"""
    if ticker not in memory.holdings or memory.holdings[ticker].get('shares', 0) == 0:
        return
    
    if ticker not in memory.alerts_sent:
        memory.alerts_sent[ticker] = {'52w_high': False}
    
    high_52w = indicators.get('52w_high', 0)
    if high_52w > 0 and abs(current_price - high_52w) <= 0.5:
        if not memory.alerts_sent[ticker]['52w_high']:
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            message = f"📈 *52-WEEK HIGH ALERT*\n"
            message += f"🔥 {symbol} reached {current_price:.2f}\n"
            message += f"📊 52W High: {high_52w:.2f}\n"
            message += f"💭 Consider SELL or HOLD decision"
            
            send_telegram_message(message)
            logger.info(f"{symbol}: 52-week high alert at {current_price:.2f}")
            memory.alerts_sent[ticker]['52w_high'] = True

# ============================
# MAIN ANALYSIS FUNCTION
# ============================

def analyze_stock(ticker: str):
    """Analyze single stock and make trading decisions"""
    try:
        logger.debug(f"Analyzing {ticker}...")
        
        historical_df = get_stock_data(ticker, period="3mo")
        if historical_df is None or historical_df.empty:
            logger.warning(f"No historical data for {ticker}")
            return
        
        indicators = calculate_indicators(historical_df)
        
        if not indicators:
            logger.warning(f"Failed to calculate indicators for {ticker}")
            return
        
        realtime_data = get_realtime_data(ticker)
        if not realtime_data:
            logger.warning(f"No real-time data for {ticker}")
            return
        
        current_price = realtime_data['price']
        atr = indicators.get('atr', 0)
        
        if atr is None:
            atr = current_price * 0.02
        
        # Update trailing stop if holding
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            update_trailing_stop(ticker, current_price, atr)
            check_52w_high_alert(ticker, current_price, indicators)
        
        # Trading decisions
        if should_sell(ticker, current_price):
            execute_sell(ticker, current_price)
        elif should_buy(ticker, indicators, current_price):
            execute_buy(ticker, current_price, indicators)
        
        # Update action status
        new_status = "HOLD" if (ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0) else "WAIT"
        memory.last_action_status[ticker] = new_status
            
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")

# ============================
# CONSOLE OUTPUT
# ============================

def print_detailed_status_table():
    """Print comprehensive status table"""
    table_data = []
    
    logger.info("Generating detailed status table...")
    
    for ticker in TICKERS:
        try:
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            
            realtime_data = get_realtime_data(ticker)
            current_price = realtime_data['price'] if realtime_data else 0.0
            
            historical_df = get_stock_data(ticker, period="3mo")
            if historical_df is not None and not historical_df.empty:
                indicators = calculate_indicators(historical_df)
            else:
                indicators = {}
            
            sma_20 = indicators.get('sma_20', 0.0) or 0.0
            sma_50 = indicators.get('sma_50', 0.0) or 0.0
            atr = indicators.get('atr', 0.0) or 0.0
            rsi = indicators.get('rsi', 0.0) or 0.0
            
            status = memory.last_action_status.get(ticker, 'WAIT')
            entry_price = 0.0
            sell_threshold = 0.0
            change_percent = 0.0
            
            if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
                entry_price = memory.holdings[ticker]['entry_price']
                sell_threshold = memory.sell_thresholds.get(ticker, 0.0)
                if entry_price > 0:
                    change_percent = ((current_price - entry_price) / entry_price) * 100
                status = 'HOLD'
            else:
                status = 'WAIT'
            
            current_price_str = f"{current_price:.2f}" if current_price > 0 else "N/A"
            entry_price_str = f"{entry_price:.2f}" if entry_price > 0 else "--"
            sma_20_str = f"{sma_20:.2f}" if sma_20 > 0 else "N/A"
            sma_50_str = f"{sma_50:.2f}" if sma_50 > 0 else "N/A"
            atr_str = f"{atr:.2f}" if atr > 0 else "N/A"
            rsi_str = f"{rsi:.1f}" if rsi > 0 else "N/A"
            sell_threshold_str = f"{sell_threshold:.2f}" if sell_threshold > 0 else "--"
            change_percent_str = f"{change_percent:+.2f}%" if change_percent != 0 else "--"
            
            table_data.append([
                symbol,
                current_price_str,
                entry_price_str,
                sma_20_str,
                sma_50_str,
                atr_str,
                rsi_str,
                sell_threshold_str,
                change_percent_str,
                status
            ])
            
        except Exception as e:
            logger.error(f"Error processing {ticker} for table: {e}")
            table_data.append([
                symbol,
                "ERROR",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "ERROR"
            ])
    
    # Print table with proper logging
    table_str = tabulate(table_data, headers=[
        "Ticker", "Current Price", "Entry Price", "20-SMA", "50-SMA",
        "ATR", "RSI", "Sell Threshold", "Change %", "Action"
    ], tablefmt="grid")
    
    logger.info("\n" + "="*120)
    logger.info("STOCK TRADING BOT - DETAILED STATUS")
    logger.info("="*120)
    logger.info(f"\n{table_str}")
    logger.info("="*120)
    
    total_positions = len([row for row in table_data if row[9] == 'HOLD'])
    waiting_positions = len([row for row in table_data if row[9] == 'WAIT'])
    logger.info(f"SUMMARY: {total_positions} HOLD | {waiting_positions} WAIT | Total P&L: {memory.total_pnl:.2f}")
    logger.info(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    logger.info("="*120)

# ============================
# TIME MANAGEMENT
# ============================

def is_market_hours() -> bool:
    """Check if market is open"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    if now.weekday() >= 5:
        logger.debug("Weekend - Market closed")
        return False
    
    is_open = MARKET_START <= current_time <= MARKET_END
    if not is_open:
        logger.debug(f"{current_time} - Market closed (Hours: {MARKET_START}-{MARKET_END})")
    
    return is_open

def is_alive_check_time() -> bool:
    """Check if it's time for alive notification"""
    current_time = datetime.now().strftime("%H:%M")
    morning_range = "09:15" <= current_time <= "09:30"
    evening_range = "15:00" <= current_time <= "15:15"
    
    return morning_range or evening_range

# ============================
# MAIN TRADING LOOP
# ============================

def main_trading_loop():
    """Main trading loop"""
    logger.info("Stock Trading Bot Started!")
    send_telegram_message("*Stock Trading Bot Started!*\n Monitoring stocks every 5 minutes")
    
    while True:
        try:
            current_time = datetime.now()
            
            # Send alive notifications
            if is_alive_check_time():
                if (memory.last_alive_check is None or 
                    (current_time - memory.last_alive_check).total_seconds() > 3600):
                    send_alive_notification()
            
            # Only trade during market hours
            if not is_market_hours():
                logger.debug(f"[{current_time.strftime('%H:%M:%S')}] Market closed. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] Analyzing stocks...")
            
            # Analyze all stocks
            for ticker in TICKERS:
                analyze_stock(ticker)
                time.sleep(1)  # Rate limiting
            
            # Print detailed status table every 15 minutes during market hours
            # if current_time.minute % 15 == 0 and current_time.second < 30:
            #     print_detailed_status_table()
            # elif is_alive_check_time():
            print_detailed_status_table()           
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] Analysis complete. Waiting {CHECK_INTERVAL//60} minutes...")
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            cleanup_and_exit()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            send_telegram_message(f"❌ *Bot Error*\nError: {str(e)}\nBot continuing...")
        
        time.sleep(CHECK_INTERVAL)

# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    # Set up exit handlers first
    setup_exit_handlers()
    
    # Verify required libraries
    try:
        import talib
        from tabulate import tabulate
        logger.info("All required libraries verified")
    except ImportError as e:
        if 'talib' in str(e):
            logger.error("TA-Lib not installed. Install with: pip install TA-Lib")
            logger.error("On Windows, you might need to download the wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        elif 'tabulate' in str(e):
            logger.error("tabulate not installed. Install with: pip install tabulate")
        sys.exit(1)
    
    # Configuration check
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logger.warning("WARNING: Telegram bot token not configured. Messages will print to console.")
    
    # Print initial status table
    logger.info("Fetching initial stock data...")
    print_detailed_status_table()
    
    # Start the trading bot
    try:
        main_trading_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        cleanup_and_exit()
    finally:
        print_final_summary()
        # print_detailed_status_table()