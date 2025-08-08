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
import os
import gc

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
# TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
# TELEGRAM_CHAT_ID = ["1012793457","1209666577"]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Trading Configuration
# TICKERS = [
#     "FILATFASH.NS", "SRESTHA.BO", "HARSHILAGR.BO", "GTLINFRA.NS", "ITC.NS",
#     "OBEROIRLTY.NS", "JAMNAAUTO.NS", "KSOLVES.NS", "ADANIGREEN.NS",
#     "TATAMOTORS.NS", "OLECTRA.NS", "ARE&M.NS", "AFFLE.NS", "BEL.NS",
#     "SUNPHARMA.NS", "LAURUSLABS.NS", "RELIANCE.NS", "KRBL.NS", "ONGC.NS",
#     "IDFCFIRSTB.NS", "BANKBARODA.NS", "GSFC.NS", "TCS.NS", "INFY.NS",
#     "SVARTCORP.BO", "SWASTIVI.BO", "BTML.NS", "SULABEN.BO", "CRYSTAL.BO",
#     "TILAK.BO", "COMFINTE.BO", "COCHINSHIP.NS", "RVNL.NS", "SHAILY.NS", "BDL.NS", 
#     "JYOTICNC.NS",  "NATIONALUM.NS", "KRONOX.NS", "SAKSOFT.NS", "ARIHANTCAP.NS",
#     "GEOJITFSL.NS", "GRAUWEIL.BO", "MCLOUD.NS", "LKPSEC.BO", "TARACHAND.NS",
#     "CENTEXT.NS", "IRISDOREME.NS", "BLIL.BO", "RNBDENIMS.BO", "ONEPOINT.NS",
#     "SONAMLTD.NS", "GATEWAY.NS", "RSYSTEMS.BO", "INDRAMEDCO.NS",
#     "JYOTHYLAB.NS", "FCL.NS", "MANINFRA.NS", "GPIL.NS", "JAGSNPHARM.NS",
#     "HSCL.NS", "JWL.NS", "BSOFT.NS", "MARKSANS.NS", "TALBROAUTO.NS",
#     "GALLANTT.NS", "RESPONIND.NS", "IRCTC.NS", "NAM-INDIA.NS", "MONARCH.NS",
#     "ELECON.NS", "SHANTIGEAR.NS", "JASH.NS", "GARFIBRES.NS", "VISHNU.NS",
#     "GRSE.NS", "RITES.NS", "AEGISLOG.NS", "ZENTEC.NS", "DELHIVERY.NS",
#     "IFCI.NS", "CDSL.NS", "NUVAMA.NS", "NEULANDLAB.NS", "GODFRYPHLP.NS",
#     "BAJAJHFL.NS", "PIDILITIND.NS", "HBLENGINE.NS", "DLF.NS", "RKFORGE.NS"
# ]

tickers_str = os.getenv("TICKERS")
TICKERS = tickers_str.split(",") if tickers_str else []

# Trading Configuration
# TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS', 
#            'BHARTIARTL.NS', 'SBIN.NS', 'LT.NS', 'HCLTECH.NS', 'WIPRO.NS']
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

# Memory Optimization Settings
BATCH_SIZE = 10  # Process 10 stocks at a time
BATCH_DELAY = 2  # 2 seconds between batches
MAX_FAILED_ATTEMPTS = 3
FAILED_TICKER_RESET_LIMIT = 20

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
        # Memory optimization tracking
        self.failed_tickers = {}  # {ticker: fail_count}
        self.batch_results = {}  # Temporary storage for batch results

memory = StockMemory()

# ============================
# MEMORY MANAGEMENT
# ============================

def cleanup_memory():
    """Aggressive memory cleanup"""
    try:
        # Clear temporary data
        memory.batch_results.clear()
        
        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collector freed {collected} objects")
        
        # Reset failed tickers if list gets too large
        if len(memory.failed_tickers) > FAILED_TICKER_RESET_LIMIT:
            logger.info(f"Resetting failed tickers list ({len(memory.failed_tickers)} entries)")
            memory.failed_tickers.clear()
            
    except Exception as e:
        logger.error(f"Error in memory cleanup: {e}")

def should_skip_ticker(ticker: str) -> bool:
    """Check if ticker should be skipped due to repeated failures"""
    return memory.failed_tickers.get(ticker, 0) >= MAX_FAILED_ATTEMPTS

def mark_ticker_failed(ticker: str):
    """Mark ticker as failed"""
    memory.failed_tickers[ticker] = memory.failed_tickers.get(ticker, 0) + 1
    if memory.failed_tickers[ticker] >= MAX_FAILED_ATTEMPTS:
        logger.warning(f"{ticker} marked as consistently failing - will skip temporarily")

def reset_ticker_failure(ticker: str):
    """Reset ticker failure count on success"""
    if ticker in memory.failed_tickers:
        del memory.failed_tickers[ticker]

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
    """Setup graceful exit handlers - only works in main thread"""
    try:
        def signal_handler(sig, frame):
            cleanup_and_exit()
        
        # Only set up signal handlers if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("Signal handlers set up successfully")
        else:
            logger.info("Skipping signal handlers - not in main thread")
            
        atexit.register(print_final_summary)
        
    except Exception as e:
        logger.warning(f"Could not set up signal handlers: {e}")
        # Still register the atexit handler
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
            f"Failed Tickers: {len(memory.failed_tickers)}",
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
    
    # chat_ids = TELEGRAM_CHAT_ID
    # .split(",") # remove when locally tested
    for chat_id in TELEGRAM_CHAT_ID:

    # chat_ids = TELEGRAM_CHAT_ID
    # for chat_id in chat_ids:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to send telegram message to {chat_id}: {response.text}")
            else:
                logger.debug(f"Telegram message sent to {chat_id}")
        except Exception as e:
            logger.error(f"Telegram error for chat {chat_id}: {e}")

def send_alive_notification():
    """Send bot alive notification"""
    current_time = datetime.now().strftime("%H:%M")
    active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
    failed_tickers_count = len(memory.failed_tickers)
    
    message = f"✅ *Stock Trading Bot is ALIVE* - {current_time}\n"
    message += f"📊 Monitoring {len(TICKERS)} stocks\n"
    message += f"💼 Active positions: {active_positions}\n"
    message += f"💰 Session P&L: {memory.total_pnl:.2f}\n"
    message += f"⚠️ Failed tickers: {failed_tickers_count}"
    
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
# OPTIMIZED DATA FETCHING
# ============================

def get_stock_data_optimized(ticker: str, period: str = "3mo", max_retries: int = 2) -> Optional[pd.DataFrame]:
    """Optimized stock data fetching with progressive retry"""
    for attempt in range(max_retries):
        try:
            # Use shorter period on retry to reduce data load
            current_period = "1mo" if attempt > 0 else period
            
            stock = yf.Ticker(ticker)
            df = stock.history(period=current_period)
            
            if df.empty:
                logger.warning(f"No data for {ticker} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
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
# TRADING LOGIC (UNCHANGED)
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
# OPTIMIZED ANALYSIS FUNCTION
# ============================

def analyze_stock_optimized(ticker: str) -> Dict:
    """Analyze single stock with memory optimization"""
    try:
        logger.debug(f"Analyzing {ticker}...")
        
        # Skip if ticker is consistently failing
        if should_skip_ticker(ticker):
            logger.debug(f"Skipping {ticker} - too many failures")
            return {
                'ticker': ticker,
                'status': 'SKIP',
                'current_price': 0.0,
                'entry_price': 0.0,
                'indicators': {},
                'error': 'Too many failures'
            }
        
        historical_df = get_stock_data_optimized(ticker, period="3mo")
        if historical_df is None or historical_df.empty:
            logger.warning(f"No historical data for {ticker}")
            mark_ticker_failed(ticker)
            return {
                'ticker': ticker,
                'status': 'ERROR',
                'current_price': 0.0,
                'entry_price': 0.0,
                'indicators': {},
                'error': 'No historical data'
            }
        
        indicators = calculate_indicators(historical_df)
        
        if not indicators:
            logger.warning(f"Failed to calculate indicators for {ticker}")
            mark_ticker_failed(ticker)
            return {
                'ticker': ticker,
                'status': 'ERROR',
                'current_price': 0.0,
                'entry_price': 0.0,
                'indicators': {},
                'error': 'Failed to calculate indicators'
            }
        
        realtime_data = get_realtime_data(ticker)
        if not realtime_data:
            logger.warning(f"No real-time data for {ticker}")
            mark_ticker_failed(ticker)
            return {
                'ticker': ticker,
                'status': 'ERROR',
                'current_price': 0.0,
                'entry_price': 0.0,
                'indicators': indicators,
                'error': 'No real-time data'
            }
        
        current_price = realtime_data['price']
        atr = indicators.get('atr', 0)
        
        if atr is None:
            atr = current_price * 0.02
        
        # Update trailing stop if holding
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            update_trailing_stop(ticker, current_price, atr)
            check_52w_high_alert(ticker, current_price, indicators)
        
        # Trading decisions
        action_taken = 'NONE'
        if should_sell(ticker, current_price):
            execute_sell(ticker, current_price)
            action_taken = 'SELL'
        elif should_buy(ticker, indicators, current_price):
            execute_buy(ticker, current_price, indicators)
            action_taken = 'BUY'
        
        # Update action status
        new_status = "HOLD" if (ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0) else "WAIT"
        memory.last_action_status[ticker] = new_status
        
        # Reset failure count on success
        reset_ticker_failure(ticker)
        
        return {
            'ticker': ticker,
            'status': new_status,
            'current_price': current_price,
            'entry_price': memory.holdings.get(ticker, {}).get('entry_price', 0.0),
            'indicators': indicators,
            'action_taken': action_taken,
            'error': None
        }
            
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        mark_ticker_failed(ticker)
        return {
            'ticker': ticker,
            'status': 'ERROR',
            'current_price': 0.0,
            'entry_price': 0.0,
            'indicators': {},
            'error': str(e)
        }

# ============================
# BATCH PROCESSING
# ============================

def process_ticker_batch(batch_tickers: List[str]) -> List[Dict]:
    """Process a batch of tickers"""
    batch_results = []
    
    logger.info(f"Processing batch of {len(batch_tickers)} stocks...")
    
    for ticker in batch_tickers:
        try:
            result = analyze_stock_optimized(ticker)
            batch_results.append(result)
            
            # Small delay between individual ticker processing
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing {ticker} in batch: {e}")
            batch_results.append({
                'ticker': ticker,
                'status': 'ERROR',
                'current_price': 0.0,
                'entry_price': 0.0,
                'indicators': {},
                'error': str(e)
            })
    
    return batch_results

def analyze_all_stocks_batched():
    """Analyze all stocks in batches with memory optimization"""
    all_results = []
    active_tickers = [ticker for ticker in TICKERS if not should_skip_ticker(ticker)]
    
    logger.info(f"Analyzing {len(active_tickers)} active tickers in batches of {BATCH_SIZE}")
    
    # Process tickers in batches
    for i in range(0, len(active_tickers), BATCH_SIZE):
        batch = active_tickers[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(active_tickers) - 1) // BATCH_SIZE + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} stocks)")
        
        try:
            batch_results = process_ticker_batch(batch)
            all_results.extend(batch_results)
            
            # Store batch results temporarily
            for result in batch_results:
                memory.batch_results[result['ticker']] = result
            
            # Memory cleanup after each batch
            cleanup_memory()
            
            # Delay between batches to avoid overwhelming the system
            if i + BATCH_SIZE < len(active_tickers):
                logger.debug(f"Waiting {BATCH_DELAY} seconds before next batch...")
                time.sleep(BATCH_DELAY)
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
    
    # Final cleanup
    cleanup_memory()
    
    logger.info(f"Completed analysis of {len(all_results)} stocks")
    return all_results

# ============================
# CONSOLE OUTPUT (OPTIMIZED)
# ============================

def print_detailed_status_table():
    """Print comprehensive status table with optimized data access"""
    table_data = []
    
    logger.info("Generating detailed status table...")
    
    for ticker in TICKERS:
        try:
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            
            # Use cached batch results if available, otherwise get fresh data
            if ticker in memory.batch_results:
                result = memory.batch_results[ticker]
                current_price = result['current_price']
                indicators = result.get('indicators', {})
                error_status = result.get('error')
            else:
                # Fallback for tickers not in batch results
                realtime_data = get_realtime_data(ticker)
                current_price = realtime_data['price'] if realtime_data else 0.0
                indicators = {}
                error_status = None
            
            # Skip failed tickers in display
            if should_skip_ticker(ticker):
                status = 'SKIP'
                current_price_str = "SKIP"
            elif error_status:
                status = 'ERROR'
                current_price_str = "ERROR"
            else:
                status = memory.last_action_status.get(ticker, 'WAIT')
                current_price_str = f"{current_price:.2f}" if current_price > 0 else "N/A"
            
            sma_20 = indicators.get('sma_20', 0.0) or 0.0
            sma_50 = indicators.get('sma_50', 0.0) or 0.0
            atr = indicators.get('atr', 0.0) or 0.0
            rsi = indicators.get('rsi', 0.0) or 0.0
            
            entry_price = 0.0
            sell_threshold = 0.0
            change_percent = 0.0
            
            if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
                entry_price = memory.holdings[ticker]['entry_price']
                sell_threshold = memory.sell_thresholds.get(ticker, 0.0)
                if entry_price > 0 and current_price > 0:
                    change_percent = ((current_price - entry_price) / entry_price) * 100
                status = 'HOLD'
            
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
            symbol = ticker.replace('.NS', '').replace('.BO', '')
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
    skipped_positions = len([row for row in table_data if row[9] == 'SKIP'])
    error_positions = len([row for row in table_data if row[9] == 'ERROR'])
    
    logger.info(f"SUMMARY: {total_positions} HOLD | {waiting_positions} WAIT | {skipped_positions} SKIP | {error_positions} ERROR")
    logger.info(f"Total P&L: {memory.total_pnl:.2f} | Failed Tickers: {len(memory.failed_tickers)}")
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
# MAIN TRADING LOOP (OPTIMIZED)
# ============================

def main_trading_loop():
    """Main trading loop with batch processing"""
    logger.info("Memory-Optimized Stock Trading Bot Started!")
    send_telegram_message("*Memory-Optimized Stock Trading Bot Started!*\n📊 Monitoring stocks in batches for better performance")
    
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
            
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] Starting batch analysis...")
            
            # Analyze all stocks in batches
            results = analyze_all_stocks_batched()
            
            # Log batch summary
            successful = len([r for r in results if r.get('error') is None])
            errors = len([r for r in results if r.get('error') is not None])
            actions = len([r for r in results if r.get('action_taken', 'NONE') != 'NONE'])
            
            logger.info(f"Batch analysis complete: {successful} successful, {errors} errors, {actions} actions taken")
            
            # Print detailed status table
            print_detailed_status_table()
            
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] Analysis complete. Waiting {CHECK_INTERVAL//60} minutes...")
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            cleanup_and_exit()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            send_telegram_message(f"❌ *Bot Error*\nError: {str(e)}\nBot continuing...")
            # Cleanup memory on error
            cleanup_memory()
        
        time.sleep(CHECK_INTERVAL)

# ============================
# ENTRY POINT
# ============================

def main():
    """Main entry point for the trading bot"""
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
    
    # Log optimization settings
    logger.info(f"Memory optimization settings:")
    logger.info(f"  - Batch size: {BATCH_SIZE} stocks")
    logger.info(f"  - Batch delay: {BATCH_DELAY} seconds")
    logger.info(f"  - Max failed attempts: {MAX_FAILED_ATTEMPTS}")
    logger.info(f"  - Failed ticker reset limit: {FAILED_TICKER_RESET_LIMIT}")
    
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

if __name__ == "__main__":
    main()
    