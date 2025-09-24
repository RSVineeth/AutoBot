import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import warnings
from datetime import datetime, timedelta
import talib
import threading
from typing import Dict, List, Optional, Tuple
from tabulate import tabulate
import atexit
import signal
import sys
import os
import logging
import psutil
import gc

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sliding_window_bot.log', mode='a', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

tickers_str = os.getenv("TICKERS")
TICKERS = tickers_str.split(",") if tickers_str else []

CHECK_INTERVAL = 60 * 5  # 5 minutes
SHARES_TO_BUY = 2
ATR_MULTIPLIER = 2.5
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
DATA_PERIOD = "3mo"

# Risk management with sliding window
TRADE_COOLDOWN_MINUTES = 30
MAX_CONSECUTIVE_LOSSES = 3
MIN_HOLDING_MINUTES = 15
MAX_DAILY_TRADES = 40

class SlidingWindowMemory:
    """
    SLIDING WINDOW DESIGN: Keep only N=3 most recent states
    - Current run data (N)
    - Previous run data (N-1) 
    - Before previous data (N-2)
    - Automatic cleanup of older data
    
    Benefits:
    - Risk management with short history
    - Minimal memory footprint 
    - Real-time data freshness
    - Pattern detection capability
    """
    
    def __init__(self, window_size=3):
        self.window_size = window_size
        
        # CORE TRADING DATA (always kept)
        self.holdings = {}
        self.sell_thresholds = {}
        self.session_start_time = datetime.now()
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        self.shutdown_flag = False
        
        # SLIDING WINDOW DATA (limited history)
        self.price_window = {}      # {ticker: [current, prev, before]}
        self.volume_window = {}     # {ticker: [current, prev, before]}
        self.rsi_window = {}        # {ticker: [current, prev, before]}
        self.signal_window = {}     # {ticker: [current, prev, before]}
        self.decision_window = {}   # {ticker: [current, prev, before]}
        
        # RISK MANAGEMENT (sliding window based)
        self.trade_outcomes = {}    # {ticker: [profit/loss, profit/loss, profit/loss]}
        self.trade_timestamps = {}  # {ticker: [datetime, datetime, datetime]}
        self.cooldown_until = {}    # {ticker: datetime}
        
        # DAILY COUNTERS
        self.daily_trades_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Initialize sliding windows for all tickers
        for ticker in TICKERS:
            self._init_ticker_windows(ticker)
    
    def _init_ticker_windows(self, ticker: str):
        """Initialize empty sliding windows for a ticker"""
        self.price_window[ticker] = [None] * self.window_size
        self.volume_window[ticker] = [None] * self.window_size
        self.rsi_window[ticker] = [None] * self.window_size
        self.signal_window[ticker] = [None] * self.window_size
        self.decision_window[ticker] = [None] * self.window_size
        self.trade_outcomes[ticker] = [None] * self.window_size
        self.trade_timestamps[ticker] = [None] * self.window_size
        self.cooldown_until[ticker] = None
    
    def slide_window(self, ticker: str, new_data: Dict):
        """
        Slide the window forward: [current, prev, before] -> [new, current, prev]
        Drop the oldest data automatically
        """
        # Slide price data
        self.price_window[ticker] = [
            new_data.get('price'),
            self.price_window[ticker][0],  # current becomes prev
            self.price_window[ticker][1]   # prev becomes before
        ]
        
        # Slide volume data
        self.volume_window[ticker] = [
            new_data.get('volume'),
            self.volume_window[ticker][0],
            self.volume_window[ticker][1]
        ]
        
        # Slide RSI data
        self.rsi_window[ticker] = [
            new_data.get('rsi'),
            self.rsi_window[ticker][0],
            self.rsi_window[ticker][1]
        ]
        
        # Slide signal strength
        self.signal_window[ticker] = [
            new_data.get('signal_strength'),
            self.signal_window[ticker][0],
            self.signal_window[ticker][1]
        ]
        
        # Slide decision data
        self.decision_window[ticker] = [
            new_data.get('decision'),
            self.decision_window[ticker][0],
            self.decision_window[ticker][1]
        ]
    
    def add_trade_outcome(self, ticker: str, profit_loss: float):
        """Add trade outcome to sliding window"""
        self.trade_outcomes[ticker] = [
            profit_loss,
            self.trade_outcomes[ticker][0],
            self.trade_outcomes[ticker][1]
        ]
        
        self.trade_timestamps[ticker] = [
            datetime.now(),
            self.trade_timestamps[ticker][0],
            self.trade_timestamps[ticker][1]
        ]
    
    def get_consecutive_losses(self, ticker: str) -> int:
        """Count recent consecutive losses using sliding window"""
        outcomes = self.trade_outcomes[ticker]
        consecutive = 0
        
        for outcome in outcomes:
            if outcome is None:
                break
            if outcome < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def is_in_cooldown(self, ticker: str) -> bool:
        """Check cooldown using sliding window logic"""
        if self.cooldown_until[ticker] is None:
            return False
        return datetime.now() < self.cooldown_until[ticker]
    
    def set_cooldown(self, ticker: str):
        """Set cooldown period"""
        self.cooldown_until[ticker] = datetime.now() + timedelta(minutes=TRADE_COOLDOWN_MINUTES)
    
    def get_trend_direction(self, ticker: str) -> str:
        """Determine trend using 3-point sliding window"""
        prices = self.price_window[ticker]
        
        # Need at least 2 valid prices
        valid_prices = [p for p in prices if p is not None]
        if len(valid_prices) < 2:
            return "UNKNOWN"
        
        # Compare current vs previous vs before
        current, prev, before = prices[0], prices[1], prices[2]
        
        if current and prev:
            if before:
                # 3-point trend
                if current > prev > before:
                    return "STRONG_UP"
                elif current < prev < before:
                    return "STRONG_DOWN"
                elif current > prev:
                    return "UP"
                elif current < prev:
                    return "DOWN"
                else:
                    return "SIDEWAYS"
            else:
                # 2-point trend
                return "UP" if current > prev else "DOWN" if current < prev else "SIDEWAYS"
        
        return "UNKNOWN"
    
    def get_momentum_change(self, ticker: str) -> str:
        """Detect momentum changes using RSI sliding window"""
        rsi_values = self.rsi_window[ticker]
        current, prev, before = rsi_values[0], rsi_values[1], rsi_values[2]
        
        if current and prev:
            rsi_change = current - prev
            if before:
                prev_change = prev - before
                # Momentum acceleration/deceleration
                if rsi_change > 5 and prev_change > 0:
                    return "ACCELERATING_UP"
                elif rsi_change < -5 and prev_change < 0:
                    return "ACCELERATING_DOWN"
            
            if rsi_change > 3:
                return "IMPROVING"
            elif rsi_change < -3:
                return "WEAKENING"
        
        return "STABLE"
    
    def should_avoid_ticker(self, ticker: str) -> Tuple[bool, str]:
        """Enhanced risk management using sliding window"""
        # Daily limit check
        if self.daily_trades_count >= MAX_DAILY_TRADES:
            return True, "Daily limit reached"
        
        # Cooldown check
        if self.is_in_cooldown(ticker):
            remaining = int((self.cooldown_until[ticker] - datetime.now()).total_seconds() / 60)
            return True, f"Cooldown ({remaining}min left)"
        
        # Consecutive losses check
        consecutive_losses = self.get_consecutive_losses(ticker)
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return True, f"Too many losses ({consecutive_losses})"
        
        # Pattern-based avoidance (using sliding window)
        trend = self.get_trend_direction(ticker)
        if trend == "STRONG_DOWN":
            return True, "Strong downtrend detected"
        
        return False, "OK"
    
    def reset_daily_counters(self):
        """Reset daily counters at start of new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades_count = 0
            self.last_reset_date = today
            logger.info("Daily counters reset")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def log_window_stats(self):
        """Log sliding window statistics"""
        total_windows = len(TICKERS) * self.window_size
        filled_slots = sum(
            sum(1 for x in window if x is not None) 
            for window in self.price_window.values()
        )
        
        logger.info(f"Sliding Window Stats: {filled_slots}/{total_windows} slots filled")
        logger.info(f"Memory Usage: {self.get_memory_usage():.1f} MB")

# Global memory instance
memory = SlidingWindowMemory()

def send_telegram_message(message: str):
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print(f"[TELEGRAM] {message}")
        return

    chat_ids = []
    if isinstance(TELEGRAM_CHAT_ID, str):
        chat_ids = [id.strip() for id in TELEGRAM_CHAT_ID.split(',')]
    elif isinstance(TELEGRAM_CHAT_ID, list):
        chat_ids = TELEGRAM_CHAT_ID
    else:
        chat_ids = [str(TELEGRAM_CHAT_ID)]

    for chat_id in chat_ids:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                print(f"Failed to send telegram message: {response.text}")
        except Exception as e:
            print(f"Telegram error: {e}")

def fetch_current_data(ticker: str) -> Optional[Dict]:
    """
    Fetch current market data - always fresh, no caching
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get minimal required data
        hist_df = stock.history(period=DATA_PERIOD)
        realtime_df = stock.history(period="2d", interval="1m")
        
        del stock  # Clear immediately
        
        if hist_df.empty or realtime_df.empty:
            return None
        
        # Extract current values
        current_price = float(realtime_df['Close'].iloc[-1])
        current_volume = float(realtime_df['Volume'].iloc[-1])
        
        # Calculate minimal indicators
        indicators = calculate_minimal_indicators(hist_df)
        
        result = {
            'price': current_price,
            'volume': current_volume,
            'rsi': indicators.get('rsi'),
            'signal_strength': calculate_signal_strength(indicators, current_price),
            'day_change': calculate_day_change(realtime_df),
            'indicators': indicators
        }
        
        # Clear dataframes immediately
        del hist_df, realtime_df
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_minimal_indicators(df: pd.DataFrame) -> Dict:
    """Calculate only essential indicators"""
    try:
        if len(df) < 50:
            return {}
        
        # Use only last 100 periods
        df_limited = df.iloc[-100:].copy()
        close = df_limited['Close'].values
        high = df_limited['High'].values
        low = df_limited['Low'].values
        volume = df_limited['Volume'].values.astype(float)
        
        indicators = {}
        
        # Essential indicators only
        sma_20 = talib.SMA(close, timeperiod=20)
        indicators['sma_20'] = float(sma_20[-1]) if len(sma_20) > 0 and not np.isnan(sma_20[-1]) else None
        
        rsi = talib.RSI(close, timeperiod=14)
        indicators['rsi'] = float(rsi[-1]) if len(rsi) > 0 and not np.isnan(rsi[-1]) else None
        
        macd, macd_signal, _ = talib.MACD(close)
        indicators['macd'] = float(macd[-1]) if len(macd) > 0 and not np.isnan(macd[-1]) else None
        indicators['macd_signal'] = float(macd_signal[-1]) if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else None
        
        atr = talib.ATR(high, low, close, timeperiod=14)
        indicators['atr'] = float(atr[-1]) if len(atr) > 0 and not np.isnan(atr[-1]) else None
        
        # Clear arrays
        del close, high, low, volume, df_limited
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

def calculate_signal_strength(indicators: Dict, current_price: float) -> float:
    """Calculate signal strength using sliding window context"""
    try:
        strength = 0.0
        
        # Basic strength from current indicators
        sma_20 = indicators.get('sma_20')
        rsi = indicators.get('rsi')
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        
        if sma_20 and current_price > sma_20:
            strength += 25
        
        if rsi and 30 < rsi < 70:
            strength += 25
        
        if macd and macd_signal and macd > macd_signal:
            strength += 25
        
        # Add trend context bonus (this is where sliding window helps)
        strength += 25  # Base strength, will be enhanced by trend analysis
        
        return min(strength, 100.0)
        
    except Exception as e:
        logger.error(f"Error calculating signal strength: {e}")
        return 0.0

def calculate_day_change(realtime_df: pd.DataFrame) -> float:
    """Calculate day change percentage"""
    try:
        if len(realtime_df) < 390:
            return 0.0
        
        day_open = float(realtime_df['Open'].iloc[-390])
        current_price = float(realtime_df['Close'].iloc[-1])
        
        return ((current_price - day_open) / day_open) * 100 if day_open > 0 else 0.0
    except:
        return 0.0

def should_buy_with_window(ticker: str, current_data: Dict) -> Tuple[bool, str]:
    """
    Make buy decision using sliding window context
    """
    try:
        # Risk management check first
        should_avoid, avoid_reason = memory.should_avoid_ticker(ticker)
        if should_avoid:
            return False, avoid_reason
        
        # Check if already holding
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            return False, "Already holding"
        
        # Current conditions
        price = current_data['price']
        rsi = current_data['rsi']
        signal_strength = current_data['signal_strength']
        indicators = current_data['indicators']
        
        # Signal strength filter
        if signal_strength < 60:
            return False, f"Signal too weak ({signal_strength:.1f})"
        
        # Trend context using sliding window
        trend = memory.get_trend_direction(ticker)
        momentum = memory.get_momentum_change(ticker)
        
        # Enhanced conditions with window context
        conditions = []
        reasons = []
        
        # 1. Basic technical conditions
        if rsi and 30 < rsi < 75:
            conditions.append(True)
            reasons.append(f"RSI good ({rsi:.1f})")
        else:
            conditions.append(False)
        
        # 2. Trend condition (enhanced with sliding window)
        if trend in ["UP", "STRONG_UP"]:
            conditions.append(True)
            reasons.append(f"Trend {trend}")
        else:
            conditions.append(False)
        
        # 3. Momentum condition
        if momentum in ["IMPROVING", "ACCELERATING_UP"]:
            conditions.append(True)
            reasons.append(f"Momentum {momentum}")
        else:
            conditions.append(False)
        
        # 4. Signal strength
        if signal_strength > 70:
            conditions.append(True)
            reasons.append(f"Strong signal")
        else:
            conditions.append(False)
        
        # Need at least 3 out of 4 conditions
        conditions_met = sum(conditions)
        if conditions_met >= 3:
            return True, f"Buy signal ({conditions_met}/4): " + ", ".join(reasons[:2])
        
        return False, f"Weak signal ({conditions_met}/4)"
        
    except Exception as e:
        logger.error(f"Error in buy decision: {e}")
        return False, "Analysis error"

def should_sell_with_window(ticker: str, current_data: Dict) -> Tuple[bool, str]:
    """
    Make sell decision using sliding window context
    """
    try:
        if ticker not in memory.holdings or memory.holdings[ticker].get('shares', 0) == 0:
            return False, "No position"
        
        price = current_data['price']
        entry_price = memory.holdings[ticker].get('entry_price', 0)
        entry_time = memory.holdings[ticker].get('entry_time', datetime.now())
        
        if entry_price == 0:
            return False, "Invalid entry price"
        
        # Parse entry_time if string
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        pnl_percent = ((price - entry_price) / entry_price) * 100
        holding_minutes = (datetime.now() - entry_time).total_seconds() / 60
        
        # 1. Stop-loss check
        if ticker in memory.sell_thresholds and price <= memory.sell_thresholds[ticker]:
            return True, f"Stop-loss hit (P&L: {pnl_percent:+.2f}%)"
        
        # 2. Minimum holding check
        if holding_minutes < MIN_HOLDING_MINUTES and pnl_percent > -5:
            return False, f"Min hold ({MIN_HOLDING_MINUTES - holding_minutes:.0f}min left)"
        
        # 3. Trend reversal using sliding window
        trend = memory.get_trend_direction(ticker)
        if trend in ["DOWN", "STRONG_DOWN"] and pnl_percent < 5:
            return True, f"Trend reversal (P&L: {pnl_percent:+.2f}%)"
        
        # 4. Take profit with momentum context
        momentum = memory.get_momentum_change(ticker)
        if pnl_percent > 8 and momentum == "WEAKENING":
            return True, f"Profit taking (P&L: {pnl_percent:+.2f}%)"
        
        # 5. Time-based exit
        if holding_minutes > 3 * 24 * 60 and pnl_percent < 3:  # 3 days, low profit
            return True, f"Time exit (P&L: {pnl_percent:+.2f}%)"
        
        return False, f"Hold (P&L: {pnl_percent:+.2f}%)"
        
    except Exception as e:
        logger.error(f"Error in sell decision: {e}")
        return False, "Analysis error"

def execute_buy_with_window(ticker: str, current_data: Dict, reason: str):
    """Execute buy and update sliding window"""
    try:
        price = current_data['price']
        indicators = current_data['indicators']
        
        # Calculate position size
        atr = indicators.get('atr', price * 0.02)
        shares = SHARES_TO_BUY
        
        # Store position
        memory.holdings[ticker] = {
            'shares': shares,
            'entry_price': price,
            'entry_time': datetime.now()
        }
        
        # Calculate stop-loss
        stop_loss = price - (ATR_MULTIPLIER * atr)
        memory.sell_thresholds[ticker] = stop_loss
        
        # Set cooldown
        memory.set_cooldown(ticker)
        
        # Update counters
        memory.total_trades += 1
        memory.daily_trades_count += 1
        
        # Send alert
        symbol = ticker.replace('.NS', '').replace('.BO', '')
        trend = memory.get_trend_direction(ticker)
        momentum = memory.get_momentum_change(ticker)
        
        message = f"🟢 *SLIDING WINDOW BUY*\n"
        message += f"📈 {symbol} - Rs.{price:.2f}\n"
        message += f"💰 Shares: {shares}\n"
        message += f"📊 Trend: {trend}\n"
        message += f"⚡ Momentum: {momentum}\n"
        message += f"🛑 Stop-loss: Rs.{stop_loss:.2f}\n"
        message += f"💡 Reason: {reason}"
        
        send_telegram_message(message)
        logger.info(f"[BUY] {symbol} @ Rs.{price:.2f} | Trend: {trend}")
        
    except Exception as e:
        logger.error(f"Error executing buy: {e}")

def execute_sell_with_window(ticker: str, current_data: Dict, reason: str):
    """Execute sell and update sliding window"""
    try:
        price = current_data['price']
        
        if ticker not in memory.holdings:
            return
        
        shares = memory.holdings[ticker].get('shares', 0)
        entry_price = memory.holdings[ticker].get('entry_price', 0)
        entry_time = memory.holdings[ticker].get('entry_time', datetime.now())
        
        if shares == 0:
            return
        
        # Parse entry_time if string
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        # Calculate P&L
        total_change = (price - entry_price) * shares
        change_percent = ((price - entry_price) / entry_price) * 100
        holding_period = datetime.now() - entry_time
        
        # Update trade outcome in sliding window
        memory.add_trade_outcome(ticker, total_change)
        
        # Set cooldown
        memory.set_cooldown(ticker)
        
        # Update statistics
        if total_change > 0:
            memory.profitable_trades += 1
        memory.total_pnl += total_change
        
        # Clear position
        del memory.holdings[ticker]
        if ticker in memory.sell_thresholds:
            del memory.sell_thresholds[ticker]
        
        # Send alert
        symbol = ticker.replace('.NS', '').replace('.BO', '')
        profit_emoji = "✅" if total_change >= 0 else "❌"
        trend = memory.get_trend_direction(ticker)
        
        message = f"🔴 *SLIDING WINDOW SELL*\n"
        message += f"📉 {symbol} - Rs.{price:.2f}\n"
        message += f"💼 Sold {shares} shares\n"
        message += f"{profit_emoji} P&L: Rs.{total_change:.2f} ({change_percent:+.2f}%)\n"
        message += f"📊 Exit Trend: {trend}\n"
        message += f"⏱️ Held: {holding_period.days}d {holding_period.seconds//3600}h\n"
        message += f"💡 Reason: {reason}"
        
        send_telegram_message(message)
        logger.info(f"[SELL] {symbol} @ Rs.{price:.2f} | P&L: Rs.{total_change:.2f}")
        
    except Exception as e:
        logger.error(f"Error executing sell: {e}")

def analyze_ticker_with_sliding_window(ticker: str):
    """
    Analyze ticker using sliding window approach
    1. Fetch current data
    2. Update sliding window
    3. Make decision with context
    4. Clear temporary data
    """
    try:
        # 1. Fetch fresh current data
        current_data = fetch_current_data(ticker)
        if not current_data:
            logger.warning(f"No data for {ticker}")
            return
        
        # 2. Update sliding window with current data
        memory.slide_window(ticker, {
            'price': current_data['price'],
            'volume': current_data['volume'],
            'rsi': current_data['rsi'],
            'signal_strength': current_data['signal_strength'],
            'decision': None  # Will be set after decision
        })
        
        # 3. Make trading decisions using window context
        should_sell, sell_reason = should_sell_with_window(ticker, current_data)
        if should_sell:
            execute_sell_with_window(ticker, current_data, sell_reason)
            memory.decision_window[ticker][0] = f"SELL: {sell_reason}"
        else:
            should_buy, buy_reason = should_buy_with_window(ticker, current_data)
            if should_buy:
                execute_buy_with_window(ticker, current_data, buy_reason)
                memory.decision_window[ticker][0] = f"BUY: {buy_reason}"
            else:
                memory.decision_window[ticker][0] = f"WAIT: {buy_reason}"
        
        # 4. Current data automatically cleared when function exits
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        memory.decision_window[ticker][0] = "ERROR"

def print_sliding_window_status():
    """Print status table with sliding window context"""
    table_data = []
    
    for ticker in TICKERS:
        try:
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            
            # Current data from sliding window
            current_price = memory.price_window[ticker][0] or 0
            current_rsi = memory.rsi_window[ticker][0] or 0
            current_decision = memory.decision_window[ticker][0] or "WAIT"
            
            # Trend and momentum from sliding window
            trend = memory.get_trend_direction(ticker)
            momentum = memory.get_momentum_change(ticker)
            consecutive_losses = memory.get_consecutive_losses(ticker)
            
            # Position data
            position = ""
            pnl = ""
            if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
                shares = memory.holdings[ticker]['shares']
                entry_price = memory.holdings[ticker]['entry_price']
                current_pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                position = f"{shares} @ Rs.{entry_price:.2f}"
                pnl = f"{current_pnl:+.2f}%"
            
            # Cooldown status
            cooldown = ""
            if memory.is_in_cooldown(ticker):
                remaining = int((memory.cooldown_until[ticker] - datetime.now()).total_seconds() / 60)
                cooldown = f"{remaining}min"
            
            table_data.append([
                symbol,
                f"Rs.{current_price:.2f}" if current_price > 0 else "N/A",
                f"{current_rsi:.1f}" if current_rsi > 0 else "N/A",
                trend[:8],  # Truncate for display
                momentum[:8],  # Truncate for display
                position or "--",
                pnl or "--",
                str(consecutive_losses) if consecutive_losses > 0 else "-",
                cooldown or "-",
                current_decision[:15] + "..." if len(current_decision) > 15 else current_decision
            ])
            
        except Exception as e:
            table_data.append([symbol, "ERROR", "N/A", "N/A", "N/A", "--", "--", "-", "-", "ERROR"])
    
    print("\n" + "="*120)
    print("SLIDING WINDOW TRADING BOT - CONTEXTUAL ANALYSIS")
    print("="*120)
    print(tabulate(table_data, headers=[
        "Ticker", "Price", "RSI", "Trend", "Momentum", "Position", "P&L%", "Losses", "Cooldown", "Decision"
    ], tablefmt="grid"))
    
    active_positions = len([t for t in memory.holdings if memory.holdings[t].get('shares', 0) > 0])
    total_cooldowns = sum(1 for ticker in TICKERS if memory.is_in_cooldown(ticker))
    
    print(f"ACTIVE POSITIONS: {active_positions}")
    print(f"COOLDOWN TICKERS: {total_cooldowns}")
    print(f"SESSION P&L: Rs.{memory.total_pnl:.2f}")
    print(f"DAILY TRADES: {memory.daily_trades_count}/{MAX_DAILY_TRADES}")
    print(f"MEMORY: {memory.get_memory_usage():.1f} MB (Sliding Window Design)")
    print("="*120)

def main_sliding_window_loop():
    """
    Main loop with sliding window approach
    - Maintains only 3 data points per ticker
    - Provides risk management context
    - Minimizes memory usage while preserving intelligence
    """
    logger.info("🤖 SLIDING WINDOW Trading Bot Started!")
    send_telegram_message("🤖 *SLIDING WINDOW Trading Bot Started!*\n✅ 3-point contextual analysis\n📊 Trend & momentum detection\n💾 Optimized memory usage\n🎯 Smart risk management")
    
    loop_count = 0
    
    while True:
        try:
            if memory.shutdown_flag:
                break
            
            loop_count += 1
            current_time = datetime.now()
            
            logger.info(f"\n[{current_time.strftime('%H:%M:%S')}] Sliding Window Cycle #{loop_count}")
            
            # Reset daily counters if new day
            memory.reset_daily_counters()
            
            # Log memory stats periodically
            if loop_count % 10 == 0:
                memory.log_window_stats()
            
            # Analyze all tickers with sliding window context
            for i, ticker in enumerate(TICKERS):
                if memory.shutdown_flag:
                    break
                
                logger.info(f"Analyzing {ticker} with sliding window ({i+1}/{len(TICKERS)})...")
                analyze_ticker_with_sliding_window(ticker)
                
                # Small delay for API rate limiting
                time.sleep(1)
            
            # Display status with sliding window context
            if loop_count % 3 == 1:  # Every 3 cycles
                print_sliding_window_status()
            
            # Send periodic update with enhanced context
            if loop_count % 12 == 0:  # Every hour
                active_positions = len([t for t in memory.holdings if memory.holdings[t].get('shares', 0) > 0])
                strong_trends = sum(1 for ticker in TICKERS if memory.get_trend_direction(ticker) in ["STRONG_UP", "UP"])
                total_cooldowns = sum(1 for ticker in TICKERS if memory.is_in_cooldown(ticker))
                
                message = f"⏰ *Sliding Window Update*\n"
                message += f"🔄 Cycle #{loop_count}\n"
                message += f"📊 Active positions: {active_positions}\n"
                message += f"📈 Uptrending: {strong_trends}/{len(TICKERS)}\n"
                message += f"⏸️ Cooldowns: {total_cooldowns}\n"
                message += f"💰 P&L: Rs.{memory.total_pnl:.2f}\n"
                message += f"🎯 Win rate: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "🎯 Win rate: 0%"
                
                send_telegram_message(message)
            
            logger.info(f"Cycle complete. Sliding windows updated. Next in {CHECK_INTERVAL//60} minutes...")
            
        except KeyboardInterrupt:
            logger.info("Sliding window bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in sliding window loop: {e}")
        
        time.sleep(CHECK_INTERVAL)

def setup_exit_handlers():
    """Setup graceful exit handlers"""
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        memory.shutdown_flag = True
        print_final_summary()
        sys.exit(0)
    
    try:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("Signal handlers set up successfully")
    except ValueError as e:
        logger.warning(f"Could not set up signal handlers: {e}")
    
    atexit.register(print_final_summary)

def print_final_summary():
    """Print final session summary"""
    try:
        session_duration = datetime.now() - memory.session_start_time
        active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
        
        print("\n" + "="*80)
        print("SLIDING WINDOW SESSION SUMMARY")
        print("="*80)
        print(f"Session Duration: {session_duration}")
        print(f"Total Trades: {memory.total_trades}")
        print(f"Profitable Trades: {memory.profitable_trades}")
        print(f"Win Rate: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "Win Rate: 0%")
        print(f"Total P&L: Rs.{memory.total_pnl:.2f}")
        print(f"Active Positions: {active_positions}")
        print(f"Memory Efficiency: {memory.get_memory_usage():.1f} MB")
        
        # Sliding window specific stats
        successful_trends = sum(1 for ticker in TICKERS 
                              if memory.get_trend_direction(ticker) in ["UP", "STRONG_UP"])
        print(f"Uptrending Tickers: {successful_trends}/{len(TICKERS)}")
        print("="*80)
    except Exception as e:
        print(f"Error in final summary: {e}")

if __name__ == "__main__":
    try:
        setup_exit_handlers()
        main_sliding_window_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Sliding window bot shutdown complete")