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
from typing import Dict, List, Optional, Tuple
from tabulate import tabulate
import atexit
import signal
import sys
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
import random

warnings.filterwarnings('ignore')

# ============================
# LOGGING CONFIGURATION
# ============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("yfinance").disabled = True

# ============================
# RATE LIMITING & CACHING
# ============================

class RateLimiter:
    def __init__(self, calls_per_second=2):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_called = 0.0
        self.lock = threading.Lock()
    
    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_called
            left_to_wait = self.min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            self.last_called = time.time()

class DataCache:
    def __init__(self, cache_duration_minutes=5):
        self.cache = {}
        self.cache_duration = cache_duration_minutes * 60
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.cache_duration:
                    return data
                else:
                    del self.cache[key]
            return None
    
    def set(self, key, data):
        with self.lock:
            self.cache[key] = (data, time.time())
    
    def clear_old(self):
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= self.cache_duration
            ]
            for key in expired_keys:
                del self.cache[key]

# Global instances
rate_limiter = RateLimiter(calls_per_second=1)  # Conservative rate limiting
data_cache = DataCache(cache_duration_minutes=3)
price_cache = DataCache(cache_duration_minutes=1)  # Shorter cache for prices

# ============================
# CONFIGURATION
# ============================

TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
TELEGRAM_CHAT_ID = ["1012793457","1209666577"]

# TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = [
    "FILATFASH.NS", "SRESTHA.BO", "HARSHILAGR.BO", "GTLINFRA.NS", "ITC.NS",
    "OBEROIRLTY.NS", "JAMNAAUTO.NS", "KSOLVES.NS", "ADANIGREEN.NS",
    "TATAMOTORS.NS", "OLECTRA.NS", "ARE&M.NS", "AFFLE.NS", "BEL.NS",
    "SUNPHARMA.NS", "LAURUSLABS.NS", "RELIANCE.NS", "KRBL.NS", "ONGC.NS",
    "IDFCFIRSTB.NS", "BANKBARODA.NS", "GSFC.NS", "TCS.NS", "INFY.NS",
    "SVARTCORP.BO", "SWASTIVI.BO", "BTML.NS", "SULABEN.BO", "CRYSTAL.BO",
    "TILAK.BO", "COMFINTE.BO", "COCHINSHIP.NS", "RVNL.NS", "SHAILY.NS", "BDL.NS", 
    "JYOTICNC.NS",  "NATIONALUM.NS", "KRONOX.NS", "SAKSOFT.NS", "ARIHANTCAP.NS",
    "GEOJITFSL.NS", "GRAUWEIL.BO", "MCLOUD.NS", "LKPSEC.BO", "TARACHAND.NS",
    "CENTEXT.NS", "IRISDOREME.NS", "BLIL.BO", "RNBDENIMS.BO", "ONEPOINT.NS",
    "SONAMLTD.NS", "GATEWAY.NS", "RSYSTEMS.BO", "INDRAMEDCO.NS",
    "JYOTHYLAB.NS", "FCL.NS", "MANINFRA.NS", "GPIL.NS", "JAGSNPHARM.NS",
    "HSCL.NS", "JWL.NS", "BSOFT.NS", "MARKSANS.NS", "TALBROAUTO.NS",
    "GALLANTT.NS", "RESPONIND.NS", "IRCTC.NS", "NAM-INDIA.NS", "MONARCH.NS",
    "ELECON.NS", "SHANTIGEAR.NS", "JASH.NS", "GARFIBRES.NS", "VISHNU.NS",
    "GRSE.NS", "RITES.NS", "AEGISLOG.NS", "ZENTEC.NS", "DELHIVERY.NS",
    "IFCI.NS", "CDSL.NS", "NUVAMA.NS", "NEULANDLAB.NS", "GODFRYPHLP.NS",
    "BAJAJHFL.NS", "PIDILITIND.NS", "HBLENGINE.NS", "DLF.NS", "RKFORGE.NS"
]

# tickers_str = os.getenv("TICKERS")
# TICKERS = tickers_str.split(",") if tickers_str else []

# Processing configuration
CHECK_INTERVAL = 60 * 5  # 5 minutes
BATCH_SIZE = 20  # Process tickers in batches
MAX_WORKERS = 5  # Concurrent threads (conservative)
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
        self.holdings = {}
        self.sell_thresholds = {}
        self.highest_prices = {}
        self.alerts_sent = {}
        self.last_action_status = {}
        self.last_alive_check = None
        self.session_start_time = datetime.now()
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        self.failed_tickers = set()  # Track tickers that consistently fail

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
    try:
        def signal_handler(sig, frame):
            cleanup_and_exit()
        
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("Signal handlers set up successfully")
        else:
            logger.info("Skipping signal handlers - not in main thread")
            
        atexit.register(print_final_summary)
        
    except Exception as e:
        logger.warning(f"Could not set up signal handlers: {e}")
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
    
    try:
        chat_ids = TELEGRAM_CHAT_ID
        # .split(",")
        for chat_id in chat_ids:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": chat_id.strip(),
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to send telegram message: {response.text}")
            else:
                logger.debug(f"Telegram message sent to {chat_id}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def send_alive_notification():
    """Send bot alive notification"""
    current_time = datetime.now().strftime("%H:%M")
    active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
    
    message = f"✅ *Stock Trading Bot is ALIVE* - {current_time}\n"
    message += f"📊 Monitoring {len(TICKERS)} stocks\n"
    message += f"💼 Active positions: {active_positions}\n"
    message += f"💰 Session P&L: {memory.total_pnl:.2f}\n"
    message += f"⚠️ Failed tickers: {len(memory.failed_tickers)}"
    
    send_telegram_message(message)
    memory.last_alive_check = datetime.now()
    logger.info(f"Alive notification sent at {current_time}")

# ============================
# OPTIMIZED DATA FETCHING
# ============================

def get_stock_data_batch(tickers: List[str], period: str = "3mo") -> Dict[str, pd.DataFrame]:
    """Fetch historical data for multiple tickers in a single call"""
    cache_key = f"batch_data_{'-'.join(sorted(tickers))}_{period}"
    cached_data = data_cache.get(cache_key)
    if cached_data is not None:
        logger.debug(f"Using cached batch data for {len(tickers)} tickers")
        return cached_data
    
    try:
        logger.info(f"Fetching batch data for {len(tickers)} tickers...")
        rate_limiter.wait()
        
        # Use yfinance download for batch processing
        data = yf.download(tickers, period=period, group_by='ticker', 
                          threads=True, progress=False)
        
        result = {}
        
        if len(tickers) == 1:
            # Single ticker case
            ticker = tickers[0]
            if not data.empty:
                result[ticker] = data
        else:
            # Multiple tickers case
            for ticker in tickers:
                try:
                    if ticker in data.columns.levels[0]:
                        ticker_data = data[ticker].dropna()
                        if not ticker_data.empty and len(ticker_data) > 50:
                            result[ticker] = ticker_data
                        else:
                            logger.warning(f"Insufficient data for {ticker}")
                            memory.failed_tickers.add(ticker)
                    else:
                        logger.warning(f"No data found for {ticker}")
                        memory.failed_tickers.add(ticker)
                except Exception as e:
                    logger.error(f"Error processing data for {ticker}: {e}")
                    memory.failed_tickers.add(ticker)
        
        data_cache.set(cache_key, result)
        logger.info(f"Batch data fetched for {len(result)} out of {len(tickers)} tickers")
        return result
        
    except Exception as e:
        logger.error(f"Error in batch data fetch: {e}")
        return {}

def get_realtime_prices_batch(tickers: List[str]) -> Dict[str, float]:
    """Get real-time prices for multiple tickers"""
    cache_key = f"batch_prices_{'-'.join(sorted(tickers))}"
    cached_prices = price_cache.get(cache_key)
    if cached_prices is not None:
        logger.debug(f"Using cached prices for {len(tickers)} tickers")
        return cached_prices
    
    try:
        logger.info(f"Fetching real-time prices for {len(tickers)} tickers...")
        rate_limiter.wait()
        
        # Create a space-separated string for yfinance
        tickers_str = ' '.join(tickers)
        data = yf.download(tickers_str, period='1d', interval='1m', 
                          threads=True, progress=False)
        
        result = {}
        
        if len(tickers) == 1:
            # Single ticker case
            if not data.empty and 'Close' in data.columns:
                result[tickers[0]] = float(data['Close'].iloc[-1])
        else:
            # Multiple tickers case
            if not data.empty and 'Close' in data.columns.levels[1]:
                for ticker in tickers:
                    try:
                        if (ticker, 'Close') in data.columns:
                            close_data = data[(ticker, 'Close')].dropna()
                            if not close_data.empty:
                                result[ticker] = float(close_data.iloc[-1])
                    except Exception as e:
                        logger.debug(f"Error getting price for {ticker}: {e}")
        
        price_cache.set(cache_key, result)
        logger.info(f"Real-time prices fetched for {len(result)} out of {len(tickers)} tickers")
        return result
        
    except Exception as e:
        logger.error(f"Error in batch price fetch: {e}")
        return {}

def get_single_stock_data(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Fallback method to get single stock data"""
    cache_key = f"single_data_{ticker}_{period}"
    cached_data = data_cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        rate_limiter.wait()
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty and len(df) > 50:
            data_cache.set(cache_key, df)
            return df
        else:
            memory.failed_tickers.add(ticker)
            return None
    except Exception as e:
        logger.error(f"Error fetching single stock data for {ticker}: {e}")
        memory.failed_tickers.add(ticker)
        return None

def get_single_realtime_price(ticker: str) -> Optional[float]:
    """Fallback method to get single stock price"""
    cache_key = f"single_price_{ticker}"
    cached_price = price_cache.get(cache_key)
    if cached_price is not None:
        return cached_price
    
    try:
        rate_limiter.wait()
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1m")
        if not df.empty:
            price = float(df['Close'].iloc[-1])
            price_cache.set(cache_key, price)
            return price
        else:
            return None
    except Exception as e:
        logger.error(f"Error getting single price for {ticker}: {e}")
        return None

# ============================
# TECHNICAL INDICATORS
# ============================

def calculate_indicators(df: pd.DataFrame) -> Dict:
    """Calculate technical indicators with error handling"""
    try:
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        volume = df['Volume'].values
        
        if len(close_prices) < 50:
            logger.warning("Not enough data for indicators calculation")
            return {}
        
        # Calculate indicators with proper error handling
        indicators = {}
        
        try:
            indicators['sma_20'] = float(talib.SMA(close_prices, timeperiod=20)[-1])
        except:
            indicators['sma_20'] = None
            
        try:
            indicators['sma_50'] = float(talib.SMA(close_prices, timeperiod=50)[-1])
        except:
            indicators['sma_50'] = None
            
        try:
            indicators['rsi'] = float(talib.RSI(close_prices, timeperiod=14)[-1])
        except:
            indicators['rsi'] = None
            
        try:
            indicators['atr'] = float(talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1])
        except:
            indicators['atr'] = None
        
        try:
            volume_sma = talib.SMA(volume.astype(float), timeperiod=20)
            if len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) and volume_sma[-1] > 0:
                indicators['volume_spike'] = volume[-1] > (volume_sma[-1] * 1.5)
            else:
                indicators['volume_spike'] = False
        except:
            indicators['volume_spike'] = False
        
        indicators['52w_high'] = float(df['High'].max())
        indicators['52w_low'] = float(df['Low'].min())
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

# ============================
# EARNINGS CHECK
# ============================

@lru_cache(maxsize=200)
def has_earnings_soon(ticker: str) -> bool:
    """Check if stock has earnings in next 2 days (cached)"""
    try:
        rate_limiter.wait()
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
# OPTIMIZED BATCH PROCESSING
# ============================

def process_ticker_batch(tickers_batch: List[str]) -> List[Dict]:
    """Process a batch of tickers efficiently"""
    results = []
    
    # Remove failed tickers from batch
    valid_tickers = [t for t in tickers_batch if t not in memory.failed_tickers]
    
    if not valid_tickers:
        return results
    
    logger.info(f"Processing batch of {len(valid_tickers)} tickers...")
    
    # Get historical data in batch
    historical_data = get_stock_data_batch(valid_tickers, period="3mo")
    
    # Get real-time prices in batch
    current_prices = get_realtime_prices_batch(valid_tickers)
    
    # Process each ticker with available data
    for ticker in valid_tickers:
        try:
            # Skip if no data available
            if ticker not in historical_data and ticker not in current_prices:
                continue
            
            # Get or fallback to single requests
            df = historical_data.get(ticker)
            if df is None:
                df = get_single_stock_data(ticker)
                if df is None:
                    continue
            
            current_price = current_prices.get(ticker)
            if current_price is None:
                current_price = get_single_realtime_price(ticker)
                if current_price is None:
                    continue
            
            # Calculate indicators
            indicators = calculate_indicators(df)
            if not indicators:
                continue
            
            # Store result
            results.append({
                'ticker': ticker,
                'price': current_price,
                'indicators': indicators,
                'data': df
            })
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            memory.failed_tickers.add(ticker)
    
    return results

def analyze_batch_results(batch_results: List[Dict]):
    """Analyze results from batch processing and make trading decisions"""
    for result in batch_results:
        try:
            ticker = result['ticker']
            current_price = result['price']
            indicators = result['indicators']
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
            logger.error(f"Error analyzing result for {result.get('ticker', 'unknown')}: {e}")

# ============================
# CONSOLE OUTPUT
# ============================

def print_detailed_status_table():
    """Print comprehensive status table"""
    table_data = []
    
    logger.info("Generating detailed status table...")
    
    # Sample a subset for display if too many tickers
    display_tickers = TICKERS[:50] if len(TICKERS) > 50 else TICKERS
    if len(TICKERS) > 50:
        logger.info(f"Displaying first 50 out of {len(TICKERS)} tickers in table")
    
    for ticker in display_tickers:
        try:
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            
            # Skip failed tickers for display
            if ticker in memory.failed_tickers:
                table_data.append([symbol, "FAILED", "--", "--", "--", "--", "--", "--", "--", "ERROR"])
                continue
            
            # Get cached or fetch data for display
            current_price = 0.0
            sma_20 = sma_50 = atr = rsi = 0.0
            
            # Try to get from cache first
            price_cache_key = f"batch_prices_{ticker}"
            cached_price = price_cache.get(price_cache_key)
            if cached_price:
                current_price = cached_price
            
            data_cache_key = f"batch_data_{ticker}_3mo"
            cached_data = data_cache.get(data_cache_key)
            if cached_data:
                indicators = calculate_indicators(cached_data)
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
                if entry_price > 0 and current_price > 0:
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
    error_positions = len([row for row in table_data if row[9] == 'ERROR'])
    
    logger.info(f"SUMMARY: {total_positions} HOLD | {waiting_positions} WAIT | {error_positions} ERROR")
    logger.info(f"Total P&L: {memory.total_pnl:.2f} | Failed tickers: {len(memory.failed_tickers)}")
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
# MAIN TRADING LOOP WITH BATCH PROCESSING
# ============================

def main_trading_loop():
    """Main trading loop with optimized batch processing"""
    logger.info("Stock Trading Bot Started!")
    logger.info(f"Monitoring {len(TICKERS)} tickers with batch processing")
    logger.info(f"Batch size: {BATCH_SIZE}, Max workers: {MAX_WORKERS}")
    send_telegram_message(f"*Stock Trading Bot Started!*\n📊 Monitoring {len(TICKERS)} stocks\n⚡ Using batch processing for efficiency")
    
    # Clean up old cache periodically
    last_cache_cleanup = time.time()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Clean up old cache every 30 minutes
            if time.time() - last_cache_cleanup > 1800:  # 30 minutes
                data_cache.clear_old()
                price_cache.clear_old()
                last_cache_cleanup = time.time()
                logger.info("Cache cleanup completed")
            
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
            
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] Starting batch analysis of {len(TICKERS)} stocks...")
            
            # Filter out failed tickers for processing
            active_tickers = [t for t in TICKERS if t not in memory.failed_tickers]
            logger.info(f"Processing {len(active_tickers)} active tickers ({len(memory.failed_tickers)} failed)")
            
            # Split tickers into batches
            ticker_batches = [active_tickers[i:i + BATCH_SIZE] for i in range(0, len(active_tickers), BATCH_SIZE)]
            
            total_processed = 0
            
            # Process each batch
            for i, batch in enumerate(ticker_batches):
                try:
                    logger.info(f"Processing batch {i+1}/{len(ticker_batches)} with {len(batch)} tickers...")
                    
                    # Process batch
                    batch_results = process_ticker_batch(batch)
                    
                    # Analyze results and make trading decisions
                    analyze_batch_results(batch_results)
                    
                    total_processed += len(batch_results)
                    
                    # Add small delay between batches to avoid rate limiting
                    if i < len(ticker_batches) - 1:  # Don't sleep after last batch
                        time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i+1}: {e}")
                    continue
            
            logger.info(f"Batch processing complete. Processed {total_processed} tickers successfully")
            
            # Print detailed status table
            print_detailed_status_table()
            
            # Remove tickers that have failed too many times
            if len(memory.failed_tickers) > len(TICKERS) * 0.3:  # If more than 30% failed
                logger.warning(f"Too many failed tickers ({len(memory.failed_tickers)}). Consider reviewing ticker list.")
            
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
# PERFORMANCE MONITORING
# ============================

def log_performance_stats():
    """Log performance statistics"""
    cache_hit_rate = (len(data_cache.cache) + len(price_cache.cache)) / max(1, len(TICKERS))
    
    logger.info("=== PERFORMANCE STATS ===")
    logger.info(f"Active tickers: {len(TICKERS) - len(memory.failed_tickers)}/{len(TICKERS)}")
    logger.info(f"Failed tickers: {len(memory.failed_tickers)}")
    logger.info(f"Data cache entries: {len(data_cache.cache)}")
    logger.info(f"Price cache entries: {len(price_cache.cache)}")
    logger.info(f"Cache efficiency: {cache_hit_rate:.2%}")
    logger.info("=" * 25)

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
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logger.warning("WARNING: Telegram bot token not configured. Messages will print to console.")
    
    if not TICKERS:
        logger.error("No tickers configured! Set TICKERS environment variable.")
        sys.exit(1)
    
    logger.info(f"Configuration loaded: {len(TICKERS)} tickers, batch size: {BATCH_SIZE}")
    
    # Print initial status table
    logger.info("Fetching initial stock data...")
    print_detailed_status_table()
    
    # Log performance stats
    log_performance_stats()
    
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