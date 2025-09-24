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
from collections import deque
import statistics
import gc
import psutil
import os
# Add after imports for persistent storage:
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('trading_bot.db')
    try:
        yield conn
    finally:
        conn.close()
warnings.filterwarnings('ignore')

# ============================
# CONFIGURATION
# ============================

# Telegram Configuration
TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
TELEGRAM_CHAT_ID = ['1012793457','1209666577']

# Trading Configuration - Extended ticker list
TICKERS = [
  "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ITC.NS","BHARTIARTL.NS","SBIN.NS","LT.NS","HCLTECH.NS","WIPRO.NS",
  "HINDUNILVR.NS","BAJFINANCE.NS","MARUTI.NS","KOTAKBANK.NS","ASIANPAINT.NS","NESTLEIND.NS","DMART.NS","BAJAJFINSV.NS","TITAN.NS","ADANIPORTS.NS",
  "AXISBANK.NS","ICICIBANK.NS","SUNPHARMA.NS","ULTRACEMCO.NS","NTPC.NS","POWERGRID.NS","TECHM.NS","M&M.NS","TATAMOTORS.NS","COALINDIA.NS",
  "JSWSTEEL.NS","TATASTEEL.NS","INDUSINDBK.NS","GRASIM.NS","DRREDDY.NS","BRITANNIA.NS","CIPLA.NS","DIVISLAB.NS","EICHERMOT.NS","HEROMOTOCO.NS",
  "HINDALCO.NS","BAJAJ-AUTO.NS","SHREE.NS","APOLLOHOSP.NS","ONGC.NS","IOC.NS","BPCL.NS","HDFCLIFE.NS","SBILIFE.NS","VEDL.NS",
  "ADANIGREEN.NS","ADANIENT.NS","ADANITRANS.NS","GODREJCP.NS","PIDILITIND.NS","DABUR.NS","MARICO.NS","COLPAL.NS","MCDOWELL-N.NS","BERGEPAINT.NS",
  "AMBUJACEM.NS","ACC.NS","SHREECEM.NS","RAMCO.NS","DALBHARAT.NS","SAIL.NS","NMDC.NS","MOIL.NS","GMRINFRA.NS","IRCTC.NS",
  "ZOMATO.NS","NYKAA.NS","PAYTM.NS","POLICYBZR.NS","TATACONSUM.NS","MUTHOOTFIN.NS","BAJAJHLDNG.NS","CHOLAFIN.NS","LTF.NS","MANAPPURAM.NS",
  "RECLTD.NS","PFC.NS","LICHSGFIN.NS","HUDCO.NS","CANBK.NS","PNB.NS","BANKBARODA.NS","UNIONBANK.NS","IDFCFIRSTB.NS","FEDERALBNK.NS",
  "RBLBANK.NS","BANDHANBNK.NS","AUBANK.NS","YESBANK.NS","IDBI.NS","IDEA.NS","BHARTI.NS","RCOM.NS","MTNL.NS","BSNL.NS",
  "RELCAPITAL.NS","RPOWER.NS","ADANIPOWER.NS","TATAPOWER.NS","NHPC.NS","FILATFASH.NS","SRESTHA.BO","HARSHILAGR.BO","GTLINFRA.NS","OBEROIRLTY.NS",
  "JAMNAAUTO.NS","KSOLVES.NS","OLECTRA.NS","ARE&M.NS","AFFLE.NS","BEL.NS","LAURUSLABS.NS","KRBL.NS","GSFC.NS","SVARTCORP.BO",
  "SWASTIVI.BO","BTML.NS","SULABEN.BO","CRYSTAL.BO","TILAK.BO","COMFINTE.BO","COCHINSHIP.NS","RVNL.NS","SHAILY.NS","BDL.NS",
  "JYOTICNC.NS","NATIONALUM.NS","KRONOX.NS","SAKSOFT.NS","ARIHANTCAP.NS","GEOJITFSL.NS","GRAUWEIL.BO","MCLOUD.NS","LKPSEC.BO","TARACHAND.NS",
  "CENTEXT.NS","IRISDOREME.NS","BLIL.BO","RNBDENIMS.BO","ONEPOINT.NS","SONAMLTD.NS","GATEWAY.NS","RSYSTEMS.BO","INDRAMEDCO.NS","JYOTHYLAB.NS",
  "FCL.NS","MANINFRA.NS","GPIL.NS","JAGSNPHARM.NS","HSCL.NS","JWL.NS","BSOFT.NS","MARKSANS.NS","TALBROAUTO.NS","GALLANTT.NS",
  "RESPONIND.NS","NAM-INDIA.NS","MONARCH.NS","ELECON.NS","SHANTIGEAR.NS","JASH.NS","GARFIBRES.NS","VISHNU.NS","GRSE.NS","RITES.NS",
  "AEGISLOG.NS","ZENTEC.NS","DELHIVERY.NS","IFCI.NS","CDSL.NS","NUVAMA.NS","NEULANDLAB.NS","GODFRYPHLP.NS","BAJAJHFL.NS","HBLENGINE.NS",
  "DLF.NS","RKFORGE.NS","EMIL.NS","JMFINANCIL.NS"
]
# [
#     'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS',
#     'BHARTIARTL.NS', 'SBIN.NS', 'LT.NS', 'HCLTECH.NS', 'WIPRO.NS',
#     'HINDUNILVR.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'KOTAKBANK.NS', 'ASIANPAINT.NS',
#     'NESTLEIND.NS', 'DMART.NS', 'BAJAJFINSV.NS', 'TITAN.NS', 'ADANIPORTS.NS',
#     'AXISBANK.NS', 'ICICIBANK.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NTPC.NS',
#     'POWERGRID.NS', 'TECHM.NS', 'M&M.NS', 'TATAMOTORS.NS', 'COALINDIA.NS',
#     'JSWSTEEL.NS', 'TATASTEEL.NS', 'INDUSINDBK.NS', 'GRASIM.NS', 'DRREDDY.NS',
#     'BRITANNIA.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS',
#     'HINDALCO.NS', 'BAJAJ-AUTO.NS', 'SHREE.NS', 'APOLLOHOSP.NS', 'ONGC.NS',
#     'IOC.NS', 'BPCL.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'VEDL.NS',
#     'ADANIGREEN.NS', 'ADANIENT.NS', 'ADANITRANS.NS', 'GODREJCP.NS', 'PIDILITIND.NS',
#     'DABUR.NS', 'MARICO.NS', 'COLPAL.NS', 'MCDOWELL-N.NS', 'BERGEPAINT.NS',
#     'AMBUJACEM.NS', 'ACC.NS', 'SHREECEM.NS', 'RAMCO.NS', 'DALBHARAT.NS',
#     'SAIL.NS', 'NMDC.NS', 'MOIL.NS', 'GMRINFRA.NS', 'IRCTC.NS',
#     'ZOMATO.NS', 'NYKAA.NS', 'PAYTM.NS', 'POLICYBZR.NS', 'TATACONSUM.NS',
#     'MUTHOOTFIN.NS', 'BAJAJHLDNG.NS', 'CHOLAFIN.NS', 'LTF.NS', 'MANAPPURAM.NS',
#     'RECLTD.NS', 'PFC.NS', 'LICHSGFIN.NS', 'HUDCO.NS', 'CANBK.NS',
#     'PNB.NS', 'BANKBARODA.NS', 'UNIONBANK.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS',
#     'RBLBANK.NS', 'BANDHANBNK.NS', 'AUBANK.NS', 'YESBANK.NS', 'IDBI.NS',
#     'IDEA.NS', 'BHARTI.NS', 'RCOM.NS', 'MTNL.NS', 'BSNL.NS',
#     'RELCAPITAL.NS', 'RPOWER.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS', 'NHPC.NS',
#     'FILATFASH.NS', 'SRESTHA.BO', 'HARSHILAGR.BO', 'GTLINFRA.NS', 'OBEROIRLTY.NS',
#     'JAMNAAUTO.NS', 'KSOLVES.NS', 'OLECTRA.NS', 'ARE&M.NS', 'AFFLE.NS',
#     'BEL.NS', 'LAURUSLABS.NS', 'KRBL.NS', 'GSFC.NS', 'SVARTCORP.BO',
#     'SWASTIVI.BO', 'BTML.NS', 'SULABEN.BO', 'CRYSTAL.BO', 'TILAK.BO',
#     'COMFINTE.BO', 'COCHINSHIP.NS', 'RVNL.NS', 'SHAILY.NS', 'BDL.NS',
#     'JYOTICNC.NS', 'NATIONALUM.NS', 'KRONOX.NS', 'SAKSOFT.NS', 'ARIHANTCAP.NS',
#     'GEOJITFSL.NS', 'GRAUWEIL.BO', 'MCLOUD.NS', 'LKPSEC.BO', 'TARACHAND.NS',
#     'CENTEXT.NS', 'IRISDOREME.NS', 'BLIL.BO', 'RNBDENIMS.BO', 'ONEPOINT.NS',
#     'SONAMLTD.NS', 'GATEWAY.NS', 'RSYSTEMS.BO', 'INDRAMEDCO.NS', 'JYOTHYLAB.NS',
#     'FCL.NS', 'MANINFRA.NS', 'GPIL.NS', 'JAGSNPHARM.NS', 'HSCL.NS',
#     'JWL.NS', 'BSOFT.NS', 'MARKSANS.NS', 'TALBROAUTO.NS', 'GALLANTT.NS',
#     'RESPONIND.NS', 'NAM-INDIA.NS', 'MONARCH.NS', 'ELECON.NS', 'SHANTIGEAR.NS',
#     'JASH.NS', 'GARFIBRES.NS', 'VISHNU.NS', 'GRSE.NS', 'RITES.NS',
#     'AEGISLOG.NS', 'ZENTEC.NS', 'DELHIVERY.NS', 'IFCI.NS', 'CDSL.NS',
#     'NUVAMA.NS', 'NEULANDLAB.NS', 'GODFRYPHLP.NS', 'BAJAJHFL.NS', 'HBLENGINE.NS',
#     'DLF.NS', 'RKFORGE.NS'
# ]

# RELAXED_MODE = True  # Add this at the top

# # Then in advanced_should_buy, use different thresholds:
# if RELAXED_MODE:
#     min_signal_strength = 50
#     required_conditions = 3  # out of 8
#     rsi_range = (25, 75)
#     volume_multiplier = 1.1
# else:
#     min_signal_strength = 70
#     required_conditions = 6
#     rsi_range = (30, 65) 
#     volume_multiplier = 1.5

CHECK_INTERVAL = 60 * 5  # 5 minutes
SHARES_TO_BUY = 2
ATR_MULTIPLIER = 1.5 # 2.0
RSI_OVERSOLD = 28 # 30 # 25
RSI_OVERBOUGHT =68 # 70 # 75

# Bulk API Configuration
BULK_FETCH_SIZE = 50  # Fetch 50 stocks at once
API_DELAY = 2  # 2 seconds between bulk calls
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Advanced Configuration
MIN_VOLUME_SPIKE = 1.2 # 1.3 # 1.8
TREND_CONFIRMATION_PERIODS = 3
VOLATILITY_FILTER = 0.03
CORRELATION_THRESHOLD = 0.7
STRENGTH_THRESHOLD = 50 # 55 # 65

# Market Hours (IST)
MARKET_START = "00:15"
MARKET_END = "23:30"
ALIVE_CHECK_MORNING = "09:15"
ALIVE_CHECK_EVENING = "15:00"

# ============================
# ENHANCED MEMORY OPTIMIZED DATA STRUCTURES
# ============================

def safe_extract(value, fallback=0.0):  # Changed default from None to 0.0
    """Safely extract numeric value from various data types"""
    if value is None:
        return fallback
    
    try:
        # Handle pandas Series
        if hasattr(value, 'iloc'):
            if len(value) > 0:
                result = float(value.iloc[-1])
                return result if not (pd.isna(result) or np.isinf(result)) else fallback
            else:
                return fallback
        
        # Handle numpy arrays
        elif hasattr(value, 'shape'):
            if value.size > 0:
                result = float(value.flat[-1])
                return result if not (np.isnan(result) or np.isinf(result)) else fallback
            else:
                return fallback
        
        # Handle lists and tuples
        elif isinstance(value, (list, tuple)):
            if len(value) > 0:
                result = float(value[-1])
                return result if not (pd.isna(result) or np.isinf(result)) else fallback
            else:
                return fallback
        
        # Handle scalar values
        else:
            result = float(value)
            return result if not (pd.isna(result) or np.isinf(result)) else fallback
            
    except (ValueError, TypeError, IndexError, AttributeError):
        return fallback
    
class EnhancedStockData:
    """Enhanced memory-efficient stock data storage"""
    def __init__(self):
        # Core position data
        self.current_positions = {}  # {ticker: {'shares': int, 'entry_price': float, 'entry_time': datetime}}
        self.stop_losses = {}  # {ticker: float}
        self.highest_prices = {}  # {ticker: float}
        self.signal_strengths = {}  # {ticker: float}
        self.last_status = {}  # {ticker: str}
        
        # Advanced tracking
        self.alerts_sent = {}  # {ticker: {'52w_high': bool, 'breakout': bool, 'support': bool}}
        self.price_history = {}  # {ticker: deque of last 20 prices}
        self.volume_history = {}  # {ticker: deque of last 20 volumes}
        self.market_sentiment = 'NEUTRAL'  # 'BULLISH', 'BEARISH', 'NEUTRAL'
        
        # Session statistics
        self.session_start = datetime.now()
        self.total_trades = 0
        self.profitable_trades = 0
        self.session_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        self.last_alive_check = None
        
        # Initialize price and volume history for all tickers
        for ticker in TICKERS:
            self.price_history[ticker] = deque(maxlen=20)
            self.volume_history[ticker] = deque(maxlen=20)
            self.alerts_sent[ticker] = {'52w_high': False, 'breakout': False, 'support': False}
            self.signal_strengths[ticker] = 0.0
# Single global instance
stock_data = EnhancedStockData()

# # def advanced_should_buy(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> Tuple[bool, str]:
# #     """Advanced buy signal with multiple confirmation criteria"""
# #     try:
# #         if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
# #             return False, "Position exists"
        
# #         # Market sentiment filter
# #         if stock_data.market_sentiment == 'BEARISH':
# #             return False, "Bearish market"
        
# #         # Get signal strength
# #         signal_strength = stock_data.signal_strengths.get(ticker, 0)
# #         if signal_strength < 50: # 55: # 70:  # Higher threshold for advanced system
# #             return False, f"Signal weak ({signal_strength:.0f})"
        
# #         # Multiple condition checks
# #         buy_conditions = []
# #         reasons = []
        
# #         # 1. Trend confirmation (Strong trend required)
# #         sma_20 = safe_extract(indicators.get('sma_20'))
# #         sma_50 = safe_extract(indicators.get('sma_50'))
# #         ema_12 = safe_extract(indicators.get('ema_12'))
# #         ema_26 = safe_extract(indicators.get('ema_26'))
        
# #         if all([sma_20, sma_50, ema_12, ema_26]):
# #             if current_price > sma_20 > sma_50 and ema_12 > ema_26:
# #                 buy_conditions.append(True)
# #                 reasons.append("Strong trend")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 2. RSI confirmation (Multiple RSI periods)
# #         rsi_14 = safe_extract(indicators.get('rsi_14'))
# #         rsi_21 = safe_extract(indicators.get('rsi_21'))
# #         if rsi_14 and rsi_21:
# #             # if 30 < rsi_14 < 65 and 35 < rsi_21 < 70:
# #             # if 25 < rsi_14 < 70 and 30 < rsi_21 < 75:
# #             if 23 < rsi_14 < 72 and 28 < rsi_21 < 78:
# #                 buy_conditions.append(True)
# #                 reasons.append(f"RSI good ({rsi_14:.1f})")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 3. MACD confirmation
# #         macd = safe_extract(indicators.get('macd'))
# #         macd_signal = safe_extract(indicators.get('macd_signal'))
# #         macd_hist = safe_extract(indicators.get('macd_histogram'))
# #         if macd and macd_signal and macd_hist:
# #             if macd > macd_signal and macd_hist > 0:
# #                 buy_conditions.append(True)
# #                 reasons.append("MACD bullish")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 4. Volume confirmation (Enhanced)
# #         volume_sma_10 = safe_extract(indicators.get('volume_sma_10'))
# #         current_volume = realtime_data.get('volume', 0)
# #         volume_roc = safe_extract(indicators.get('volume_roc'))
# #         if volume_sma_10 and current_volume:
# #             volume_condition = current_volume > volume_sma_10 # * 1.2 # 1.5
# #             if volume_roc and volume_roc > 8: # 10: # 15:  # Additional volume momentum
# #                 volume_condition = True
# #             if volume_condition:
# #                 buy_conditions.append(True)
# #                 reasons.append("Volume surge")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 5. Stochastic confirmation
# #         stoch_k = safe_extract(indicators.get('stoch_k'))
# #         stoch_d = safe_extract(indicators.get('stoch_d'))
# #         if stoch_k and stoch_d:
# #             if stoch_k > stoch_d and stoch_k < 82: # 80: # 75:
# #                 buy_conditions.append(True)
# #                 reasons.append("Stoch bullish")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 6. Bollinger Bands position
# #         bb_lower = safe_extract(indicators.get('bb_lower'))
# #         bb_upper = safe_extract(indicators.get('bb_upper'))
# #         bb_middle = safe_extract(indicators.get('bb_middle'))
# #         if bb_lower and bb_upper and bb_middle:
# #             # Good position: above middle but not near upper band
# #             # if bb_middle < current_price < bb_upper * 0.95:
# #             # if bb_middle * 1.02 < current_price < bb_upper * 0.98:
# #             if bb_middle * 1.04 < current_price < bb_upper * 0.96:
# #                 buy_conditions.append(True)
# #                 reasons.append("BB position good")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 7. Volatility filter
# #         volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
# #         if volatility_ratio and volatility_ratio < 2.8: #2.5: # 2.0:
# #             buy_conditions.append(True)
# #             reasons.append("Low volatility")
# #         else:
# #             buy_conditions.append(False)
        
# #         # 8. Support level check
# #         support_1 = safe_extract(indicators.get('support_1'))
# #         if support_1 and current_price > support_1: # * 1.01: # 1.03:
# #             buy_conditions.append(True)
# #             reasons.append("Above support")
# #         else:
# #             buy_conditions.append(False)
        
# #         # Need at least 6 out of 8 conditions for advanced buy signal
# #         conditions_met = sum(buy_conditions)
# #         if conditions_met >= 4: # 6:
# #             return True, f"Advanced buy ({conditions_met}/8): " + ", ".join(reasons[:4])
        
# #         return False, f"Insufficient conditions ({conditions_met}/8)"
        
# #     except Exception as e:
# #         print(f"Error in advanced_should_buy for {ticker}: {e}")
# #         return False, "Error in analysis"

# # MAIN ISSUES AND FIXES FOR TRADING BOT ALERTS


# # ============================
# # ISSUE 1: SIGNAL STRENGTH THRESHOLDS - MODERATELY STRICT
# # ============================

# def advanced_should_buy(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> Tuple[bool, str]:
#     """Advanced buy signal with moderately strict confirmation criteria"""
#     try:
#         if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
#             return False, "Position exists"
        
#         # Market sentiment filter - MODERATELY STRICT
#         if stock_data.market_sentiment == 'BEARISH':
#             # Allow some trades in bearish market but be more selective
#             signal_strength_required = 50  # Higher threshold in bearish market
#         else:
#             signal_strength_required = 40  # Normal threshold
        
#         # Get signal strength - MODERATELY STRICT THRESHOLD
#         signal_strength = stock_data.signal_strengths.get(ticker, 0)
#         if signal_strength < signal_strength_required:
#             return False, f"Signal weak ({signal_strength:.0f})"
        
#         # Multiple condition checks - MODERATELY STRICT REQUIREMENTS
#         buy_conditions = []
#         reasons = []
        
#         # 1. Trend confirmation (Moderately Strict)
#         sma_20 = safe_extract(indicators.get('sma_20'))
#         sma_50 = safe_extract(indicators.get('sma_50'))
#         ema_12 = safe_extract(indicators.get('ema_12'))
#         ema_26 = safe_extract(indicators.get('ema_26'))
        
#         if all([sma_20, sma_50, ema_12, ema_26]):
#             # Require price above SMA20 AND either SMA20>SMA50 OR strong EMA trend
#             if current_price > sma_20 and (sma_20 > sma_50 * 0.998 or ema_12 > ema_26 * 1.005):
#                 buy_conditions.append(True)
#                 reasons.append("Trend good")
#             else:
#                 buy_conditions.append(False)
        
#         # 2. RSI confirmation (STRICTER RANGES)
#         rsi_14 = safe_extract(indicators.get('rsi_14'))
#         rsi_21 = safe_extract(indicators.get('rsi_21'))
#         if rsi_14 and rsi_21:
#             if 25 < rsi_14 < 70 and 30 < rsi_21 < 75:  # Tightened ranges
#                 buy_conditions.append(True)
#                 reasons.append(f"RSI good ({rsi_14:.1f})")
#             else:
#                 buy_conditions.append(False)
        
#         # 3. MACD confirmation (STRICTER)
#         macd = safe_extract(indicators.get('macd'))
#         macd_signal = safe_extract(indicators.get('macd_signal'))
#         macd_hist = safe_extract(indicators.get('macd_histogram'))
#         if macd and macd_signal:
#             # Require MACD > signal AND positive momentum (or very close)
#             if macd > macd_signal and (not macd_hist or macd_hist > -0.01):
#                 buy_conditions.append(True)
#                 reasons.append("MACD bullish")
#             else:
#                 buy_conditions.append(False)
        
#         # 4. Volume confirmation (STRICTER)
#         volume_sma_10 = safe_extract(indicators.get('volume_sma_10'))
#         current_volume = realtime_data.get('volume', 0)
#         if volume_sma_10 and current_volume:
#             if current_volume > volume_sma_10 * 1.0:  # At least average volume
#                 buy_conditions.append(True)
#                 reasons.append("Volume adequate")
#             else:
#                 buy_conditions.append(False)
        
#         # 5. Stochastic confirmation (STRICTER)
#         stoch_k = safe_extract(indicators.get('stoch_k'))
#         stoch_d = safe_extract(indicators.get('stoch_d'))
#         if stoch_k and stoch_d:
#             if stoch_k < 80 and stoch_k > stoch_d * 0.95:  # Not overbought and showing momentum
#                 buy_conditions.append(True)
#                 reasons.append("Stoch good")
#             else:
#                 buy_conditions.append(False)
        
#         # 6. Bollinger Bands position (STRICTER)
#         bb_lower = safe_extract(indicators.get('bb_lower'))
#         bb_upper = safe_extract(indicators.get('bb_upper'))
#         bb_middle = safe_extract(indicators.get('bb_middle'))
#         if bb_lower and bb_upper and bb_middle:
#             # Require price above middle band but not too close to upper
#             if bb_middle * 1.01 < current_price < bb_upper * 0.95:
#                 buy_conditions.append(True)
#                 reasons.append("BB position good")
#             else:
#                 buy_conditions.append(False)
        
#         # 7. Volatility filter (STRICTER)
#         volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
#         if volatility_ratio is None or volatility_ratio < 3.0:  # More strict volatility limit
#             buy_conditions.append(True)
#             reasons.append("Volatility OK")
#         else:
#             buy_conditions.append(False)
        
#         # 8. Support level check (STRICTER)
#         support_1 = safe_extract(indicators.get('support_1'))
#         if support_1 is None or current_price > support_1 * 1.02:  # Need to be clearly above support
#             buy_conditions.append(True)
#             reasons.append("Above support")
#         else:
#             buy_conditions.append(False)
        
#         # CHANGED: Need 4 out of 8 conditions (was 3) - MORE STRICT
#         conditions_met = sum(buy_conditions)
#         if conditions_met >= 4:
#             return True, f"Quality buy ({conditions_met}/8): " + ", ".join(reasons[:4])
        
#         return False, f"Insufficient conditions ({conditions_met}/8)"
        
#     except Exception as e:
#         print(f"Error in advanced_should_buy for {ticker}: {e}")
#         return False, "Error in analysis"

# # ============================
# # ISSUE 2: SIGNAL STRENGTH CALCULATION - MODERATELY STRICT
# # ============================

# def calculate_advanced_signal_strength(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> float:
#     """MODERATELY STRICT signal strength calculation"""
#     try:
#         if not indicators:
#             return 25.0  # Lower base score
        
#         strength_components = []
        
#         # === TREND STRENGTH (30 points) - STRICTER ===
#         sma_20 = safe_extract(indicators.get('sma_20'))
#         sma_50 = safe_extract(indicators.get('sma_50'))
#         ema_12 = safe_extract(indicators.get('ema_12'))
#         ema_26 = safe_extract(indicators.get('ema_26'))
        
#         trend_score = 5  # Lower base score
#         if sma_20 and current_price > sma_20 * 1.01:  # Need to be clearly above
#             trend_score += 10
#         if sma_50 and current_price > sma_50 * 1.02:  # Even more clearly above
#             trend_score += 8
#         if ema_12 and ema_26 and ema_12 > ema_26 * 1.005:  # Strong EMA trend
#             trend_score += 7
        
#         strength_components.append(min(trend_score, 30))
        
#         # === MOMENTUM STRENGTH (25 points) - STRICTER ===
#         rsi = safe_extract(indicators.get('rsi_14'))
#         macd = safe_extract(indicators.get('macd'))
#         macd_signal = safe_extract(indicators.get('macd_signal'))
        
#         momentum_score = 3  # Lower base score
#         if rsi and 30 < rsi < 65:  # Tighter sweet spot
#             momentum_score += 12
#         elif rsi and 25 < rsi < 70:  # Acceptable range
#             momentum_score += 8
#         elif rsi and 20 < rsi < 75:  # Wider but lower score
#             momentum_score += 5
        
#         if macd and macd_signal and macd > macd_signal * 1.02:  # Strong MACD signal
#             momentum_score += 10
#         elif macd and macd_signal and macd > macd_signal:  # Basic MACD signal
#             momentum_score += 6
        
#         strength_components.append(min(momentum_score, 25))
        
#         # === VOLUME STRENGTH (20 points) - STRICTER ===
#         volume_sma_10 = safe_extract(indicators.get('volume_sma_10'))
#         current_volume = realtime_data.get('volume', 0)
        
#         volume_score = 2  # Lower base score
#         if volume_sma_10 and current_volume:
#             if current_volume > volume_sma_10 * 1.5:  # Strong volume
#                 volume_score += 12
#             elif current_volume > volume_sma_10 * 1.2:  # Good volume
#                 volume_score += 8
#             elif current_volume > volume_sma_10:  # Average volume
#                 volume_score += 4
#         else:
#             volume_score += 3  # Small credit if no volume data
        
#         strength_components.append(min(volume_score, 20))
        
#         # === OTHER INDICATORS (25 points) - STRICTER ===
#         other_score = 8  # Lower base score
        
#         # Bollinger position - more strict
#         bb_lower = safe_extract(indicators.get('bb_lower'))
#         bb_upper = safe_extract(indicators.get('bb_upper'))
#         bb_middle = safe_extract(indicators.get('bb_middle'))
#         if bb_lower and bb_upper and bb_middle:
#             if bb_middle < current_price < bb_upper * 0.95:  # Sweet spot
#                 other_score += 8
#             elif bb_lower < current_price < bb_upper:  # Acceptable
#                 other_score += 4
        
#         # Support check - stricter
#         support_1 = safe_extract(indicators.get('support_1'))
#         if support_1 and current_price > support_1 * 1.05:  # Well above support
#             other_score += 6
#         elif not support_1 or current_price > support_1:  # Basic support
#             other_score += 3
        
#         # Additional quality check
#         if bb_middle and sma_20 and abs(current_price - bb_middle) < abs(current_price - sma_20):
#             other_score += 3  # Bonus for good BB positioning
        
#         strength_components.append(min(other_score, 25))
        
#         # Calculate total with smaller bonus
#         total_strength = sum(strength_components)
#         return min(total_strength + 5, 100.0)  # Reduced bonus from 10 to 5
        
#     except Exception as e:
#         print(f"Error calculating signal strength for {ticker}: {e}")
#         return 30.0  # Return moderate base score on error


# # # ============================
# # # ISSUE 1: SIGNAL STRENGTH THRESHOLDS ARE TOO HIGH
# # # ============================

# # # FIX: Lower the threshold to get more signals
# # def advanced_should_buy(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> Tuple[bool, str]:
# #     """Advanced buy signal with multiple confirmation criteria"""
# #     try:
# #         if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
# #             return False, "Position exists"
        
# #         # Market sentiment filter - RELAXED
# #         if stock_data.market_sentiment == 'BEARISH':
# #             # CHANGE: Don't block all trades in bearish market
# #             pass  # Just log but don't block
        
# #         # Get signal strength - LOWERED THRESHOLD
# #         signal_strength = stock_data.signal_strengths.get(ticker, 0)
# #         if signal_strength < 35:  # CHANGED FROM 50 to 35
# #             return False, f"Signal weak ({signal_strength:.0f})"
        
# #         # Multiple condition checks - RELAXED REQUIREMENTS
# #         buy_conditions = []
# #         reasons = []
        
# #         # 1. Trend confirmation (Relaxed)
# #         sma_20 = safe_extract(indicators.get('sma_20'))
# #         sma_50 = safe_extract(indicators.get('sma_50'))
# #         ema_12 = safe_extract(indicators.get('ema_12'))
# #         ema_26 = safe_extract(indicators.get('ema_26'))
        
# #         if all([sma_20, sma_50, ema_12, ema_26]):
# #             if current_price > sma_20 and ema_12 > ema_26:  # RELAXED: Removed sma_20 > sma_50
# #                 buy_conditions.append(True)
# #                 reasons.append("Trend OK")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 2. RSI confirmation (RELAXED RANGES)
# #         rsi_14 = safe_extract(indicators.get('rsi_14'))
# #         rsi_21 = safe_extract(indicators.get('rsi_21'))
# #         if rsi_14 and rsi_21:
# #             if 20 < rsi_14 < 80 and 25 < rsi_21 < 85:  # MUCH MORE RELAXED
# #                 buy_conditions.append(True)
# #                 reasons.append(f"RSI OK ({rsi_14:.1f})")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 3. MACD confirmation (RELAXED)
# #         macd = safe_extract(indicators.get('macd'))
# #         macd_signal = safe_extract(indicators.get('macd_signal'))
# #         if macd and macd_signal:
# #             if macd > macd_signal:  # REMOVED macd_hist > 0 requirement
# #                 buy_conditions.append(True)
# #                 reasons.append("MACD bullish")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 4. Volume confirmation (RELAXED)
# #         volume_sma_10 = safe_extract(indicators.get('volume_sma_10'))
# #         current_volume = realtime_data.get('volume', 0)
# #         if volume_sma_10 and current_volume:
# #             if current_volume > volume_sma_10 * 0.8:  # MUCH LOWER THRESHOLD
# #                 buy_conditions.append(True)
# #                 reasons.append("Volume OK")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 5. Stochastic confirmation (RELAXED)
# #         stoch_k = safe_extract(indicators.get('stoch_k'))
# #         stoch_d = safe_extract(indicators.get('stoch_d'))
# #         if stoch_k and stoch_d:
# #             if stoch_k < 90:  # VERY RELAXED - just not extremely overbought
# #                 buy_conditions.append(True)
# #                 reasons.append("Stoch OK")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 6. Bollinger Bands position (RELAXED)
# #         bb_lower = safe_extract(indicators.get('bb_lower'))
# #         bb_upper = safe_extract(indicators.get('bb_upper'))
# #         bb_middle = safe_extract(indicators.get('bb_middle'))
# #         if bb_lower and bb_upper and bb_middle:
# #             if current_price > bb_lower:  # VERY RELAXED - just above lower band
# #                 buy_conditions.append(True)
# #                 reasons.append("BB position OK")
# #             else:
# #                 buy_conditions.append(False)
        
# #         # 7. Volatility filter (RELAXED)
# #         volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
# #         if volatility_ratio is None or volatility_ratio < 5.0:  # VERY RELAXED
# #             buy_conditions.append(True)
# #             reasons.append("Volatility OK")
# #         else:
# #             buy_conditions.append(False)
        
# #         # 8. Support level check (RELAXED)
# #         support_1 = safe_extract(indicators.get('support_1'))
# #         if support_1 is None or current_price > support_1 * 0.95:  # RELAXED
# #             buy_conditions.append(True)
# #             reasons.append("Above support")
# #         else:
# #             buy_conditions.append(False)
        
# #         # CHANGED: Need only 3 out of 8 conditions (was 4)
# #         conditions_met = sum(buy_conditions)
# #         if conditions_met >= 3:
# #             return True, f"Relaxed buy ({conditions_met}/8): " + ", ".join(reasons[:4])
        
# #         return False, f"Insufficient conditions ({conditions_met}/8)"
        
# #     except Exception as e:
# #         print(f"Error in advanced_should_buy for {ticker}: {e}")
# #         return False, "Error in analysis"

# # # ============================
# # # ISSUE 2: SIGNAL STRENGTH CALCULATION IS TOO STRICT
# # # ============================

# # def calculate_advanced_signal_strength(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> float:
# #     """RELAXED signal strength calculation"""
# #     try:
# #         if not indicators:
# #             return 30.0  # Give some base score
        
# #         strength_components = []
        
# #         # === TREND STRENGTH (30 points) - RELAXED ===
# #         sma_20 = safe_extract(indicators.get('sma_20'))
# #         sma_50 = safe_extract(indicators.get('sma_50'))
# #         ema_12 = safe_extract(indicators.get('ema_12'))
# #         ema_26 = safe_extract(indicators.get('ema_26'))
        
# #         trend_score = 10  # Start with base score
# #         if sma_20 and current_price > sma_20:
# #             trend_score += 8
# #         if sma_50 and current_price > sma_50:
# #             trend_score += 5
# #         if ema_12 and ema_26 and ema_12 > ema_26:
# #             trend_score += 7
        
# #         strength_components.append(min(trend_score, 30))
        
# #         # === MOMENTUM STRENGTH (25 points) - RELAXED ===
# #         rsi = safe_extract(indicators.get('rsi_14'))
# #         macd = safe_extract(indicators.get('macd'))
# #         macd_signal = safe_extract(indicators.get('macd_signal'))
        
# #         momentum_score = 8  # Start with base score
# #         if rsi and 25 < rsi < 75:  # Much wider range
# #             momentum_score += 10
# #         elif rsi and 20 < rsi < 80:
# #             momentum_score += 7
        
# #         if macd and macd_signal and macd > macd_signal:
# #             momentum_score += 7
        
# #         strength_components.append(min(momentum_score, 25))
        
# #         # === VOLUME STRENGTH (20 points) - RELAXED ===
# #         volume_sma_10 = safe_extract(indicators.get('volume_sma_10'))
# #         current_volume = realtime_data.get('volume', 0)
        
# #         volume_score = 5  # Start with base score
# #         if volume_sma_10 and current_volume:
# #             if current_volume > volume_sma_10 * 1.1:
# #                 volume_score += 10
# #             elif current_volume > volume_sma_10 * 0.8:
# #                 volume_score += 5
# #         else:
# #             volume_score += 5  # Give credit if no volume data
        
# #         strength_components.append(min(volume_score, 20))
        
# #         # === OTHER INDICATORS (25 points) - RELAXED ===
# #         other_score = 15  # Start with good base score
        
# #         # Bollinger position
# #         bb_lower = safe_extract(indicators.get('bb_lower'))
# #         bb_upper = safe_extract(indicators.get('bb_upper'))
# #         if bb_lower and bb_upper and bb_lower < current_price < bb_upper:
# #             other_score += 5
        
# #         # Support check
# #         support_1 = safe_extract(indicators.get('support_1'))
# #         if not support_1 or current_price > support_1:
# #             other_score += 5
        
# #         strength_components.append(min(other_score, 25))
        
# #         # Calculate total with bonus for having data
# #         total_strength = sum(strength_components)
# #         return min(total_strength + 10, 100.0)  # Add 10 point bonus
        
# #     except Exception as e:
# #         print(f"Error calculating signal strength for {ticker}: {e}")
# #         return 40.0  # Return decent base score on error


# ============================
# ISSUE 1: SIGNAL STRENGTH THRESHOLDS - VERY SLIGHTLY MORE STRICT
# ============================

def advanced_should_buy(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> Tuple[bool, str]:
    """Advanced buy signal with very slightly more strict confirmation criteria"""
    try:
        if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
            return False, "Position exists"
        
        # Market sentiment filter - VERY SLIGHTLY MORE STRICT
        if stock_data.market_sentiment == 'BEARISH':
            # Require higher threshold in bearish market
            signal_strength_required = 53  # Increased from 52
        else:
            signal_strength_required = 43  # Increased from 42
        
        # Get signal strength - VERY SLIGHTLY HIGHER THRESHOLD
        signal_strength = stock_data.signal_strengths.get(ticker, 0)
        if signal_strength < signal_strength_required:
            return False, f"Signal weak ({signal_strength:.0f})"
        
        # Multiple condition checks - VERY SLIGHTLY MORE STRICT REQUIREMENTS
        buy_conditions = []
        reasons = []
        
        # 1. Trend confirmation (Very Slightly More Strict)
        sma_20 = safe_extract(indicators.get('sma_20'))
        sma_50 = safe_extract(indicators.get('sma_50'))
        ema_12 = safe_extract(indicators.get('ema_12'))
        ema_26 = safe_extract(indicators.get('ema_26'))
        
        if all([sma_20, sma_50, ema_12, ema_26]):
            # Require price above SMA20 AND either SMA20>SMA50 OR strong EMA trend
            if current_price > sma_20 and (sma_20 > sma_50 * 1.0005 or ema_12 > ema_26 * 1.007):  # Very slightly tighter
                buy_conditions.append(True)
                reasons.append("Trend good")
            else:
                buy_conditions.append(False)
        
        # 2. RSI confirmation (VERY SLIGHTLY STRICTER RANGES)
        rsi_14 = safe_extract(indicators.get('rsi_14'))
        rsi_21 = safe_extract(indicators.get('rsi_21'))
        if rsi_14 and rsi_21:
            if 28 < rsi_14 < 67 and 33 < rsi_21 < 72:  # Very slightly tightened ranges
                buy_conditions.append(True)
                reasons.append(f"RSI good ({rsi_14:.1f})")
            else:
                buy_conditions.append(False)
        
        # 3. MACD confirmation (VERY SLIGHTLY STRICTER)
        macd = safe_extract(indicators.get('macd'))
        macd_signal = safe_extract(indicators.get('macd_signal'))
        macd_hist = safe_extract(indicators.get('macd_histogram'))
        if macd and macd_signal:
            # Require MACD > signal AND positive momentum (or very close)
            if macd > macd_signal * 1.002 and (not macd_hist or macd_hist > -0.007):  # Very slightly tighter
                buy_conditions.append(True)
                reasons.append("MACD bullish")
            else:
                buy_conditions.append(False)
        
        # 4. Volume confirmation (VERY SLIGHTLY STRICTER)
        volume_sma_10 = safe_extract(indicators.get('volume_sma_10'))
        current_volume = realtime_data.get('volume', 0)
        if volume_sma_10 and current_volume:
            if current_volume > volume_sma_10 * 1.06:  # Very slightly increased from 1.05
                buy_conditions.append(True)
                reasons.append("Volume adequate")
            else:
                buy_conditions.append(False)
        
        # 5. Stochastic confirmation (VERY SLIGHTLY STRICTER)
        stoch_k = safe_extract(indicators.get('stoch_k'))
        stoch_d = safe_extract(indicators.get('stoch_d'))
        if stoch_k and stoch_d:
            if stoch_k < 77 and stoch_k > stoch_d * 0.975:  # Very slightly more strict
                buy_conditions.append(True)
                reasons.append("Stoch good")
            else:
                buy_conditions.append(False)
        
        # 6. Bollinger Bands position (VERY SLIGHTLY STRICTER)
        bb_lower = safe_extract(indicators.get('bb_lower'))
        bb_upper = safe_extract(indicators.get('bb_upper'))
        bb_middle = safe_extract(indicators.get('bb_middle'))
        if bb_lower and bb_upper and bb_middle:
            # Require price above middle band but not too close to upper
            if bb_middle * 1.018 < current_price < bb_upper * 0.935:  # Very slightly tighter range
                buy_conditions.append(True)
                reasons.append("BB position good")
            else:
                buy_conditions.append(False)
        
        # 7. Volatility filter (VERY SLIGHTLY STRICTER)
        volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
        if volatility_ratio is None or volatility_ratio < 2.75:  # Very slightly reduced from 2.8
            buy_conditions.append(True)
            reasons.append("Volatility OK")
        else:
            buy_conditions.append(False)
        
        # 8. Support level check (VERY SLIGHTLY STRICTER)
        support_1 = safe_extract(indicators.get('support_1'))
        if support_1 is None or current_price > support_1 * 1.027:  # Very slightly increased from 1.025
            buy_conditions.append(True)
            reasons.append("Above support")
        else:
            buy_conditions.append(False)
        
        # Still need 4 out of 8 conditions (maintaining strictness)
        conditions_met = sum(buy_conditions)
        if conditions_met >= 4:
            return True, f"Quality buy ({conditions_met}/8): " + ", ".join(reasons[:4])
        
        return False, f"Insufficient conditions ({conditions_met}/8)"
        
    except Exception as e:
        print(f"Error in advanced_should_buy for {ticker}: {e}")
        return False, "Error in analysis"

# ============================
# ISSUE 2: SIGNAL STRENGTH CALCULATION - VERY SLIGHTLY MORE STRICT
# ============================

def calculate_advanced_signal_strength(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> float:
    """VERY SLIGHTLY MORE STRICT signal strength calculation"""
    try:
        if not indicators:
            return 21.0  # Very slightly reduced from 22.0
        
        strength_components = []
        
        # === TREND STRENGTH (30 points) - VERY SLIGHTLY STRICTER ===
        sma_20 = safe_extract(indicators.get('sma_20'))
        sma_50 = safe_extract(indicators.get('sma_50'))
        ema_12 = safe_extract(indicators.get('ema_12'))
        ema_26 = safe_extract(indicators.get('ema_26'))
        
        trend_score = 3  # Very slightly reduced from 4
        if sma_20 and current_price > sma_20 * 1.013:  # Very slightly increased from 1.012
            trend_score += 10
        if sma_50 and current_price > sma_50 * 1.023:  # Very slightly increased from 1.022
            trend_score += 8
        if ema_12 and ema_26 and ema_12 > ema_26 * 1.007:  # Very slightly increased from 1.006
            trend_score += 7
        
        strength_components.append(min(trend_score, 30))
        
        # === MOMENTUM STRENGTH (25 points) - VERY SLIGHTLY STRICTER ===
        rsi = safe_extract(indicators.get('rsi_14'))
        macd = safe_extract(indicators.get('macd'))
        macd_signal = safe_extract(indicators.get('macd_signal'))
        
        momentum_score = 1  # Very slightly reduced from 2
        if rsi and 35 < rsi < 60:  # Very slightly tighter sweet spot
            momentum_score += 12
        elif rsi and 30 < rsi < 65:  # Very slightly tighter acceptable range
            momentum_score += 8
        elif rsi and 25 < rsi < 70:  # Very slightly tighter wider range
            momentum_score += 5
        
        if macd and macd_signal and macd > macd_signal * 1.027:  # Very slightly increased from 1.025
            momentum_score += 10
        elif macd and macd_signal and macd > macd_signal * 1.003:  # Very slightly stricter basic signal
            momentum_score += 6
        
        strength_components.append(min(momentum_score, 25))
        
        # === VOLUME STRENGTH (20 points) - VERY SLIGHTLY STRICTER ===
        volume_sma_10 = safe_extract(indicators.get('volume_sma_10'))
        current_volume = realtime_data.get('volume', 0)
        
        volume_score = 0  # Very slightly reduced from 1
        if volume_sma_10 and current_volume:
            if current_volume > volume_sma_10 * 1.65:  # Very slightly increased from 1.6
                volume_score += 12
            elif current_volume > volume_sma_10 * 1.35:  # More strict for 5-6% reduction
                volume_score += 8
            elif current_volume > volume_sma_10 * 1.06:  # Very slightly increased from 1.05
                volume_score += 4
        else:
            volume_score += 1  # Very slightly reduced credit from 2 to 1
        
        strength_components.append(min(volume_score, 20))
        
        # === OTHER INDICATORS (25 points) - VERY SLIGHTLY STRICTER ===
        other_score = 6  # Very slightly reduced from 7
        
        # Bollinger position - very slightly more strict
        bb_lower = safe_extract(indicators.get('bb_lower'))
        bb_upper = safe_extract(indicators.get('bb_upper'))
        bb_middle = safe_extract(indicators.get('bb_middle'))
        if bb_lower and bb_upper and bb_middle:
            if bb_middle * 1.008 < current_price < bb_upper * 0.925:  # Very slightly tighter sweet spot
                other_score += 8
            elif bb_lower < current_price < bb_upper * 0.955:  # Very slightly tighter acceptable range
                other_score += 4
        
        # Support check - very slightly stricter
        support_1 = safe_extract(indicators.get('support_1'))
        if support_1 and current_price > support_1 * 1.065:  # Very slightly increased from 1.06
            other_score += 6
        elif not support_1 or current_price > support_1 * 1.012:  # Very slightly stricter basic support
            other_score += 3
        
        # Additional quality check - very slightly stricter
        if bb_middle and sma_20 and abs(current_price - bb_middle) < abs(current_price - sma_20) * 0.94:
            other_score += 3  # Very slightly tighter bonus condition
        
        strength_components.append(min(other_score, 25))
        
        # Calculate total with smaller bonus
        total_strength = sum(strength_components)
        return min(total_strength + 2, 100.0)  # Very slightly reduced bonus from 3 to 2
        
    except Exception as e:
        print(f"Error calculating signal strength for {ticker}: {e}")
        return 26.0  # Very slightly reduced error fallback from 27.0

# ============================
# ISSUE 3: ADD DEBUG LOGGING TO SEE WHAT'S HAPPENING
# ============================

def bulk_analyze_stocks_advanced():
    """Enhanced bulk analysis with DEBUG LOGGING"""
    try:
        print(f"\n🔥 Starting advanced bulk analysis of {len(TICKERS)} stocks...")
        
        # Bulk fetch historical data
        print("📊 Fetching historical data...")
        historical_data = bulk_fetch_stock_data(TICKERS, period="6mo")
        
        # Force garbage collection
        gc.collect()
        
        # Bulk fetch real-time data
        print("⚡ Fetching real-time data...")
        realtime_data = bulk_fetch_realtime_data(TICKERS)
        
        print(f"✅ Data fetched: {len(historical_data)} historical, {len(realtime_data)} real-time")
        
        # Process each stock with advanced analysis
        analysis_results = []
        buy_signals_found = 0
        sell_signals_found = 0
        
        for ticker in TICKERS:
            try:
                symbol = ticker.replace('.NS', '')
                
                # Get data
                hist_df = historical_data.get(ticker)
                rt_data = realtime_data.get(ticker)
                
                if hist_df is None or rt_data is None:
                    analysis_results.append([symbol, "NO DATA", "N/A", "--", "--", "--", "--", "0", "--", "--", "--", "ERROR"])
                    continue
                
                current_price = rt_data['price']
                day_change = rt_data.get('day_change', 0.0)
                
                # Calculate advanced indicators
                indicators = calculate_advanced_indicators(hist_df)
                
                # Calculate advanced signal strength
                signal_strength = calculate_advanced_signal_strength(ticker, indicators, current_price, rt_data)
                stock_data.signal_strengths[ticker] = signal_strength
                
                # DEBUG: Print signal strength for first 5 stocks
                if len(analysis_results) < 5:
                    print(f"DEBUG {symbol}: Signal strength = {signal_strength:.1f}")
                
                # Update dynamic trailing stops for existing positions
                if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
                    update_dynamic_trailing_stop(ticker, current_price, indicators)
                
                # Check for advanced alerts
                check_advanced_alerts(ticker, current_price, indicators, rt_data)
                
                # Advanced trading decisions
                should_sell, sell_reason = advanced_should_sell(ticker, indicators, current_price)
                if should_sell:
                    execute_advanced_sell(ticker, current_price, sell_reason)
                    stock_data.last_status[ticker] = 'SELL_SIGNAL'
                    sell_signals_found += 1
                    print(f"🔴 SELL SIGNAL: {symbol} - {sell_reason}")
                else:
                    should_buy, buy_reason = advanced_should_buy(ticker, indicators, current_price, rt_data)
                    if should_buy:
                        execute_advanced_buy(ticker, current_price, indicators, buy_reason)
                        stock_data.last_status[ticker] = 'BUY_SIGNAL'
                        buy_signals_found += 1
                        print(f"🟢 BUY SIGNAL: {symbol} - {buy_reason}")
                    else:
                        if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
                            stock_data.last_status[ticker] = f'HOLD'
                        else:
                            stock_data.last_status[ticker] = f'WAIT'
                            # DEBUG: Print why no buy signal for first few stocks
                            if len(analysis_results) < 3:
                                print(f"DEBUG {symbol}: No buy - {buy_reason}")
                
                # [Rest of the table preparation code remains the same...]
                # Add table row building here (after the buy/sell logic)
                symbol = ticker.replace('.NS', '')
                price_str = f"₹{current_price:.2f}"
                day_change_str = f"{day_change:+.1f}%" if 'day_change' in locals() else "N/A"

                # Get indicator values for display
                rsi_val = safe_extract(indicators.get('rsi_14'))
                macd_val = safe_extract(indicators.get('macd'))
                stoch_val = safe_extract(indicators.get('stoch_k'))
                bb_pos = "N/A"  # Calculate BB position if needed

                # Get position info if exists
                if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
                    entry_price = stock_data.current_positions[ticker]['entry_price']
                    shares = stock_data.current_positions[ticker]['shares']
                    current_pnl = ((current_price - entry_price) / entry_price) * 100
                    entry_str = f"₹{entry_price:.2f}"
                    pnl_str = f"{current_pnl:+.1f}%"
                    value_str = f"₹{(current_price * shares):.0f}"
                    stop_str = f"₹{stock_data.stop_losses.get(ticker, 0):.2f}"
                else:
                    entry_str = "--"
                    pnl_str = "--"
                    value_str = "--"
                    stop_str = "--"

                # Signal strength display
                signal_strength = stock_data.signal_strengths.get(ticker, 0)
                if signal_strength > 75:
                    signal_display = f"🟢 {signal_strength:.0f}"
                elif signal_strength > 50:
                    signal_display = f"🟡 {signal_strength:.0f}"
                else:
                    signal_display = f"🔴 {signal_strength:.0f}"

                status = stock_data.last_status.get(ticker, 'WAIT')

                analysis_results.append([
                    symbol, price_str, day_change_str, entry_str, 
                    f"{rsi_val:.1f}" if rsi_val else "--",
                    f"{macd_val:.3f}" if macd_val else "--",
                    f"{stoch_val:.1f}" if stoch_val else "--",
                    bb_pos, signal_display, stop_str, pnl_str, value_str, status
                ])
            except Exception as e:
                symbol = ticker.replace('.NS', '')
                print(f"Error analyzing {symbol}: {e}")
                analysis_results.append([symbol, "ERROR", "N/A", "--", "--", "--", "--", "--", "0", "--", "--", "--", "ERROR"])
        
        # Update market sentiment
        stock_data.market_sentiment = calculate_market_sentiment()
        
        # DEBUG: Print summary
        print(f"DEBUG: Found {buy_signals_found} buy signals, {sell_signals_found} sell signals")
        print(f"DEBUG: Market sentiment: {stock_data.market_sentiment}")
        
        # Print enhanced results table
        print_advanced_analysis_table(analysis_results)
        
        # Force cleanup
        del historical_data
        del realtime_data
        gc.collect()
        
        return len(analysis_results)

    except Exception as e:
        print(f"Error in advanced bulk analysis: {e}")
        return 0

# ============================
# ISSUE 4: FORCE SEND TEST ALERT
# ============================

# def send_test_alert():
#     """Send a test alert to verify Telegram is working"""
#     test_message = f"🧪 *TEST ALERT*\n"
#     test_message += f"📱 Telegram connection working!\n"
#     test_message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
#     test_message += f"🤖 Bot is alive and monitoring\n"
#     test_message += f"🎯 Looking for trading signals..."
    
#     send_telegram_message(test_message)
#     print("Test alert sent!")

# ============================
# TEMPORARY FIX: FORCE GENERATE SOME ALERTS
# ============================

def force_generate_test_buy_signal():
    """Temporarily force a buy signal for testing"""
    if len(TICKERS) > 0:
        test_ticker = TICKERS[0]  # Use first ticker
        symbol = test_ticker.replace('.NS', '')
        
        message = f"🟢 *TEST BUY SIGNAL*\n"
        message += f"📈 {symbol} @ ₹100.00\n"
        message += f"💰 Qty: 2 shares\n"
        message += f"🎯 Signal: 85/100 (STRONG)\n"
        message += f"📊 RSI: 45.2 | ATR: ₹2.50\n"
        message += f"🛡 Stop: ₹95.00\n"
        message += f"💡 Test signal for debugging\n"
        message += f"📈 Market: NEUTRAL"
        
        send_telegram_message(message)
        print("Test buy signal sent!")

# Single global instance
stock_data = EnhancedStockData()
def advanced_should_sell(ticker: str, indicators: Dict, current_price: float) -> Tuple[bool, str]:
    """Advanced sell signal with dynamic conditions"""
    if ticker not in stock_data.current_positions or stock_data.current_positions[ticker].get('shares', 0) == 0:
        return False, "No position"
    
    try:
        entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
        current_pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # Dynamic stop-loss check
        if ticker in stock_data.stop_losses and current_price <= stock_data.stop_losses[ticker]:
            return True, f"Dynamic stop-loss (PnL: {current_pnl:+.2f}%)"
        
        # Advanced profit-taking conditions
        if current_pnl > 5: # 10:  # 10% profit threshold
            sell_conditions = []
            
            # RSI overbought
            rsi = safe_extract(indicators.get('rsi_14'))
            if rsi and rsi > 70: # 75:
                sell_conditions.append("RSI overbought")
            
            # Stochastic overbought
            stoch_k = safe_extract(indicators.get('stoch_k'))
            if stoch_k and stoch_k > 75: # 80:
                sell_conditions.append("Stoch overbought")
            
            # MACD divergence
            macd = safe_extract(indicators.get('macd'))
            macd_signal = safe_extract(indicators.get('macd_signal'))
            if macd and macd_signal and macd < macd_signal:
                sell_conditions.append("MACD bearish")
            
            if len(sell_conditions) >= 2:
                return True, f"Profit-take (PnL: {current_pnl:+.2f}%): " + ", ".join(sell_conditions[:2])
        
        # Trend reversal detection
        sma_20 = safe_extract(indicators.get('sma_20'))
        ema_12 = safe_extract(indicators.get('ema_12'))
        ema_26 = safe_extract(indicators.get('ema_26'))
        
        if all([sma_20, ema_12, ema_26]):
            if current_price < sma_20 * 0.97 and ema_12 < ema_26:
                return True, f"Trend reversal (PnL: {current_pnl:+.2f}%)"
        
        # Bollinger Band squeeze exit
        bb_upper = safe_extract(indicators.get('bb_upper'))
        bb_lower = safe_extract(indicators.get('bb_lower'))
        if bb_upper and bb_lower and current_price > bb_upper:
            volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
            if volatility_ratio and volatility_ratio > 2.0: # 2.5:  # High volatility
                return True, f"Volatility exit (PnL: {current_pnl:+.2f}%)"
        
        # Time-based exit with profit conditions
        if ticker in stock_data.current_positions and 'entry_time' in stock_data.current_positions[ticker]:
            holding_time = datetime.now() - stock_data.current_positions[ticker]['entry_time']
            if holding_time.days > 7 and current_pnl < 3:  # 7 days with minimal profit
                return True, f"Time exit (PnL: {current_pnl:+.2f}%, {holding_time.days}d)"
        
        return False, f"Hold (PnL: {current_pnl:+.2f}%)"
        
    except Exception as e:
        print(f"Error in advanced_should_sell for {ticker}: {e}")
        return False, "Error in sell analysis"

def execute_advanced_buy(ticker: str, current_price: float, indicators: Dict, reason: str):
    """Execute advanced buy with dynamic position sizing"""
    try:
        atr = safe_extract(indicators.get('atr_14'))
        if atr is None or atr <= 0:
            atr = current_price * 0.02
        
        # Dynamic position sizing based on volatility and signal strength
        volatility_ratio = safe_extract(indicators.get('volatility_ratio'), 1.0)
        signal_strength = stock_data.signal_strengths.get(ticker, 0)
        
        # Adjust shares based on signal strength and volatility
        base_shares = SHARES_TO_BUY
        if signal_strength > 85:
            base_shares = int(SHARES_TO_BUY * 1.5)  # Increase position for very strong signals
        elif signal_strength > 75:
            base_shares = int(SHARES_TO_BUY * 1.2)
        
        adjusted_shares = max(1, int(base_shares / volatility_ratio))
        
        stock_data.current_positions[ticker] = {
            'shares': adjusted_shares,
            'entry_price': current_price,
            'entry_time': datetime.now()
        }
        
        # Dynamic stop-loss calculation
        support_level = safe_extract(indicators.get('support_1'), current_price * 0.94)
        atr_stop = current_price - (ATR_MULTIPLIER * atr)
        bb_lower = safe_extract(indicators.get('bb_lower'))
        
        # Use the highest of support, ATR stop, or BB lower band
        dynamic_stops = [support_level, atr_stop]
        if bb_lower:
            dynamic_stops.append(bb_lower * 0.98)
        
        dynamic_stop = max(dynamic_stops)
        stock_data.stop_losses[ticker] = dynamic_stop
        stock_data.highest_prices[ticker] = current_price
        
        stock_data.total_trades += 1
        
        # Enhanced buy notification
        symbol = ticker.replace('.NS', '')
        rsi_val = safe_extract(indicators.get('rsi_14'))
        signal_strength = stock_data.signal_strengths.get(ticker, 0)
        
        message = f"🟢 *ADVANCED BUY*\n"
        message += f"📈 {symbol} @ ₹{current_price:.2f}\n"
        message += f"💰 Qty: {adjusted_shares} (Dynamic)\n"
        message += f"🎯 Signal: {signal_strength:.1f}/100\n"
        message += f"📊 RSI: {rsi_val:.1f} | ATR: ₹{atr:.2f}\n"
        message += f"🛑 Smart Stop: ₹{dynamic_stop:.2f}\n"
        message += f"💡 {reason}\n"
        message += f"📈 Market: {stock_data.market_sentiment}"
        
        send_telegram_message(message)
        print(f"[ADVANCED BUY] {symbol} @ ₹{current_price:.2f} | Qty: {adjusted_shares} | Signal: {signal_strength:.1f}")
        
    except Exception as e:
        print(f"Buy execution error for {ticker}: {e}")

def execute_advanced_sell(ticker: str, current_price: float, reason: str):
    """Execute advanced sell with detailed tracking"""
    try:
        if ticker not in stock_data.current_positions:
            return
        
        shares = stock_data.current_positions[ticker].get('shares', 0)
        entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
        entry_time = stock_data.current_positions[ticker].get('entry_time', datetime.now())
        
        if shares == 0:
            return
        
        # Calculate detailed P&L
        total_change = (current_price - entry_price) * shares
        change_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        holding_period = datetime.now() - entry_time
        
        # Update session statistics
        stock_data.session_pnl += total_change
        if total_change > 0:
            stock_data.profitable_trades += 1
        
        # Update drawdown tracking
        current_portfolio_value = stock_data.session_pnl
        if current_portfolio_value > stock_data.peak_portfolio_value:
            stock_data.peak_portfolio_value = current_portfolio_value
        else:
            drawdown = ((stock_data.peak_portfolio_value - current_portfolio_value) / stock_data.peak_portfolio_value) * 100
            if drawdown > stock_data.max_drawdown:
                stock_data.max_drawdown = drawdown
        
        # Clear position and reset alerts
        stock_data.current_positions[ticker] = {'shares': 0, 'entry_price': 0}
        if ticker in stock_data.stop_losses:
            del stock_data.stop_losses[ticker]
        if ticker in stock_data.highest_prices:
            del stock_data.highest_prices[ticker]
        
        # Reset alerts
        stock_data.alerts_sent[ticker] = {'52w_high': False, 'breakout': False, 'support': False}
        
        symbol = ticker.replace('.NS', '')
        profit_emoji = "💚" if total_change >= 0 else "❌"
        holding_days = holding_period.days
        holding_hours = holding_period.seconds // 3600
        
        message = f"🔴 *ADVANCED SELL*\n"
        message += f"📉 {symbol} @ ₹{current_price:.2f}\n"
        message += f"💼 Qty: {shares}\n"
        message += f"{profit_emoji} P&L: ₹{total_change:.2f} ({change_percent:+.2f}%)\n"
        message += f"⏱️ Held: {holding_days}d {holding_hours}h\n"
        message += f"💡 {reason}\n"
        message += f"📊 Session P&L: ₹{stock_data.session_pnl:.2f}"
        
        send_telegram_message(message)
        print(f"[ADVANCED SELL] {symbol} @ ₹{current_price:.2f} | P&L: ₹{total_change:.2f}")
        
    except Exception as e:
        print(f"Sell execution error for {ticker}: {e}")

def update_dynamic_trailing_stop(ticker: str, current_price: float, indicators: Dict):
    """Update dynamic trailing stop-loss with ATR and support levels"""
    if ticker not in stock_data.current_positions or stock_data.current_positions[ticker].get('shares', 0) == 0:
        return
    
    try:
        atr = safe_extract(indicators.get('atr_14'))
        if atr is None or atr <= 0:
            atr = current_price * 0.02
        
        # Update highest price
        if ticker not in stock_data.highest_prices:
            stock_data.highest_prices[ticker] = current_price
        else:
            stock_data.highest_prices[ticker] = max(stock_data.highest_prices[ticker], current_price)
        
        # Dynamic ATR multiplier based on profit
        entry_price = stock_data.current_positions[ticker].get('entry_price', current_price)
        profit_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # Tighten stop-loss as profit increases
        if profit_percent > 15:
            atr_multiplier = 1.2  # Very tight stop for high profits
        elif profit_percent > 10:
            atr_multiplier = 1.5  # Tight stop
        elif profit_percent > 5:
            atr_multiplier = 1.8  # Medium stop
        else:
            atr_multiplier = ATR_MULTIPLIER  # Standard stop
        
        # Calculate new trailing stop
        new_stop = stock_data.highest_prices[ticker] - (atr_multiplier * atr)
        
        # Support level consideration
        support_level = safe_extract(indicators.get('support_1'))
        if support_level and support_level < new_stop:
            new_stop = max(new_stop, support_level * 0.98)  # Small buffer below support
        
        # Bollinger lower band consideration
        bb_lower = safe_extract(indicators.get('bb_lower'))
        if bb_lower and bb_lower > entry_price:  # Only if BB lower is above entry
            new_stop = max(new_stop, bb_lower * 0.98)
        
        # Only update if new stop is higher (trailing up)
        if ticker not in stock_data.stop_losses:
            stock_data.stop_losses[ticker] = new_stop
        else:
            stock_data.stop_losses[ticker] = max(stock_data.stop_losses[ticker], new_stop)
            
    except Exception as e:
        print(f"Error updating trailing stop for {ticker}: {e}")

def check_advanced_alerts(ticker: str, current_price: float, indicators: Dict, realtime_data: Dict):
    """Check for advanced alerts and notifications"""
    try:
        symbol = ticker.replace('.NS', '')
        
        # 1. Breakout Alert
        if not stock_data.alerts_sent[ticker]['breakout']:
            bb_upper = safe_extract(indicators.get('bb_upper'))
            volume_sma = safe_extract(indicators.get('volume_sma_10'))
            current_volume = realtime_data.get('volume', 0)
            rsi = safe_extract(indicators.get('rsi_14'))
            
            # Enhanced breakout conditions
            if all([bb_upper, volume_sma, rsi]) and current_price > bb_upper:
                volume_spike = current_volume > volume_sma * 2 if volume_sma > 0 else False
                rsi_condition = 50 < rsi < 80  # Not overbought breakout
                
                if volume_spike and rsi_condition:
                    message = f"🚀 *BREAKOUT ALERT*\n"
                    message += f"📈 {symbol} - Bullish breakout!\n"
                    message += f"💰 Price: ₹{current_price:.2f} vs BB: ₹{bb_upper:.2f}\n"
                    message += f"📊 Volume: {(current_volume/volume_sma):.1f}x avg\n"
                    message += f"🎯 RSI: {rsi:.1f} (Healthy breakout)\n"
                    message += f"⚡ Consider position entry"
                    
                    send_telegram_message(message)
                    stock_data.alerts_sent[ticker]['breakout'] = True
        
        # 2. Support Level Alert (for holdings)
        if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
            if not stock_data.alerts_sent[ticker]['support']:
                support_level = safe_extract(indicators.get('support_1'))
                bb_lower = safe_extract(indicators.get('bb_lower'))
                
                support_alerts = []
                if support_level and current_price < support_level * 1.03:
                    support_alerts.append(f"Key support: ₹{support_level:.2f}")
                
                if bb_lower and current_price < bb_lower * 1.02:
                    support_alerts.append(f"BB lower: ₹{bb_lower:.2f}")
                
                if support_alerts:
                    entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
                    pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    
                    message = f"⚠️ *SUPPORT ALERT*\n"
                    message += f"📉 {symbol} near support levels\n"
                    message += f"💰 Current: ₹{current_price:.2f}\n"
                    message += f"🛡️ {' | '.join(support_alerts)}\n"
                    message += f"📊 Your P&L: {pnl:+.2f}%\n"
                    message += f"👀 Watch for bounce or exit"
                    
                    send_telegram_message(message)
                    stock_data.alerts_sent[ticker]['support'] = True
        
        # 3. 52-Week High Alert (for holdings)
        if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
            if not stock_data.alerts_sent[ticker]['52w_high']:
                high_52w = indicators.get('52w_high', 0)
                distance_from_high = safe_extract(indicators.get('distance_from_52w_high'))
                
                if distance_from_high and distance_from_high < 3:  # Within 3% of 52w high
                    entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
                    profit = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    rsi = safe_extract(indicators.get('rsi_14'))
                    
                    message = f"🏆 *52W HIGH ALERT*\n"
                    message += f"🚀 {symbol} near yearly peak!\n"
                    message += f"📊 Current: ₹{current_price:.2f}\n"
                    message += f"🎯 52W High: ₹{high_52w:.2f}\n"
                    message += f"💚 Your gain: {profit:+.2f}%\n"
                    if rsi:
                        message += f"📈 RSI: {rsi:.1f}\n"
                    message += f"🤔 Consider profit booking"
                    
                    send_telegram_message(message)
                    stock_data.alerts_sent[ticker]['52w_high'] = True
        
        # 4. Volume Surge Alert (for watchlist)
        volume_sma_30 = safe_extract(indicators.get('volume_sma_30'))
        current_volume = realtime_data.get('volume', 0)
        if volume_sma_30 and current_volume > volume_sma_30 * 3:  # 3x volume surge
            # Only send if not already in position and signal is strong
            if (ticker not in stock_data.current_positions or 
                stock_data.current_positions[ticker].get('shares', 0) == 0):
                signal_strength = stock_data.signal_strengths.get(ticker, 0)
                if signal_strength > 60:
                    day_change = realtime_data.get('day_change', 0)
                    message = f"📢 *VOLUME SURGE*\n"
                    message += f"⚡ {symbol} - Unusual activity\n"
                    message += f"💰 Price: ₹{current_price:.2f} ({day_change:+.2f}%)\n"
                    message += f"📊 Volume: {(current_volume/volume_sma_30):.1f}x normal\n"
                    message += f"🎯 Signal: {signal_strength:.0f}/100\n"
                    message += f"👁️ Worth monitoring"
                    
                    # Don't spam - limit to strong signals only
                    if signal_strength > 70:
                        send_telegram_message(message)
                    
    except Exception as e:
        print(f"Error checking alerts for {ticker}: {e}")

def print_advanced_analysis_table(analysis_results):
    """Print enhanced analysis results table"""
    try:
        # Calculate enhanced summary statistics
        total_positions = len([row for row in analysis_results if "₹" in row[11] and row[11] != "--"])
        waiting_positions = len(TICKERS) - total_positions
        strong_signals = len([row for row in analysis_results if "🟢" in row[8]])
        medium_signals = len([row for row in analysis_results if "🟡" in row[8]])
        weak_signals = len([row for row in analysis_results if "🔴" in row[8]])
        errors = len([row for row in analysis_results if row[1] == "ERROR"])
        
        # Calculate win rate and average signal strength
        win_rate = (stock_data.profitable_trades / stock_data.total_trades * 100) if stock_data.total_trades > 0 else 0
        avg_signal_strength = sum(stock_data.signal_strengths.values()) / len(stock_data.signal_strengths) if stock_data.signal_strengths else 0
        
        print("\n" + "="*160)
        print("ENHANCED BULK TRADING BOT - ADVANCED TECHNICAL ANALYSIS")
        print("="*160)
        print(tabulate(analysis_results, headers=[
            "Stock", "Price", "Day%", "Entry", "RSI", "MACD", "Stoch", "BB%", 
            "Signal", "Stop", "P&L%", "Value", "Status"
        ], tablefmt="grid"))
        print("="*160)
        
        print(f"📊 ANALYSIS: {len(analysis_results)} stocks | {errors} errors | Avg Signal: {avg_signal_strength:.1f}")
        print(f"💼 POSITIONS: {total_positions} active | {waiting_positions} waiting")
        print(f"🎯 SIGNALS: {strong_signals} strong 🟢 | {medium_signals} medium 🟡 | {weak_signals} weak 🔴")
        print(f"📈 MARKET SENTIMENT: {stock_data.market_sentiment}")
        print(f"💰 SESSION P&L: ₹{stock_data.session_pnl:.2f} | MAX DRAWDOWN: {stock_data.max_drawdown:.2f}%")
        print(f"🏆 WIN RATE: {win_rate:.1f}% ({stock_data.profitable_trades}/{stock_data.total_trades})")
        print(f"⏰ LAST UPDATED: {datetime.now().strftime('%H:%M:%S')}")
        print("="*160)
        
    except Exception as e:
        print(f"Error printing advanced table: {e}")

# ============================
# ENHANCED TELEGRAM FUNCTIONS
# ============================

def send_telegram_message(message: str):
    """Send enhanced message to Telegram with error handling"""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print(f"[TELEGRAM] {message}")
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
                print(f"Telegram failed: {response.status_code}")
        except Exception as e:
            print(f"Telegram error: {e}")

def send_enhanced_alive_notification():
    """Send enhanced bot alive notification"""
    current_time = datetime.now().strftime("%H:%M")
    active_positions = sum(1 for pos in stock_data.current_positions.values() if pos.get('shares', 0) > 0)
    strong_signals = sum(1 for strength in stock_data.signal_strengths.values() if strength > 75)
    medium_signals = sum(1 for strength in stock_data.signal_strengths.values() if 60 <= strength <= 75)
    
    # Calculate average signal strength
    avg_signal = sum(stock_data.signal_strengths.values()) / len(stock_data.signal_strengths) if stock_data.signal_strengths else 0
    
    # Calculate session performance
    win_rate = (stock_data.profitable_trades / stock_data.total_trades * 100) if stock_data.total_trades > 0 else 0
    
    message = f"✅ *Enhanced Bulk Trading Bot ALIVE* - {current_time}\n"
    message += f"📊 Monitoring {len(TICKERS)} stocks with 15+ indicators\n"
    message += f"💼 Active positions: {active_positions}\n"
    message += f"🎯 Signals: {strong_signals} strong | {medium_signals} medium\n"
    message += f"📈 Market sentiment: {stock_data.market_sentiment}\n"
    message += f"⚡ Avg signal strength: {avg_signal:.1f}/100\n"
    message += f"💰 Session P&L: ₹{stock_data.session_pnl:.2f}\n"
    message += f"🏆 Win rate: {win_rate:.1f}%\n"
    message += f"🚀 Advanced features: Dynamic stops, Smart alerts, Volume analysis"
    
    send_telegram_message(message)
    stock_data.last_alive_check = datetime.now()

def send_hourly_summary():
    """Send detailed hourly summary"""
    active_positions = sum(1 for pos in stock_data.current_positions.values() if pos.get('shares', 0) > 0)
    strong_signals = sum(1 for strength in stock_data.signal_strengths.values() if strength > 75)
    
    # Get top performing stocks
    top_performers = []
    for ticker in TICKERS:
        if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
            entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
            if ticker in stock_data.price_history and len(stock_data.price_history[ticker]) > 0:
                current_price = stock_data.price_history[ticker][-1]
                pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                top_performers.append((ticker.replace('.NS', ''), pnl))
    
    # Sort by performance
    top_performers.sort(key=lambda x: x[1], reverse=True)
    
    # Session duration
    session_duration = datetime.now() - stock_data.session_start
    hours = int(session_duration.total_seconds() // 3600)
    minutes = int((session_duration.total_seconds() % 3600) // 60)
    
    message = f"📊 *Hourly Trading Summary*\n"
    message += f"⏰ Session: {hours}h {minutes}m\n"
    message += f"💼 Active: {active_positions} | Strong signals: {strong_signals}\n"
    message += f"📈 Market: {stock_data.market_sentiment} | P&L: ₹{stock_data.session_pnl:.2f}\n"
    
    if stock_data.total_trades > 0:
        win_rate = (stock_data.profitable_trades / stock_data.total_trades) * 100
        message += f"🎯 Trades: {stock_data.total_trades} | Win rate: {win_rate:.1f}%\n"
    
    if top_performers:
        message += f"🏆 Top performers:\n"
        for symbol, pnl in top_performers[:3]:  # Top 3
            message += f"   • {symbol}: {pnl:+.2f}%\n"
    
    message += f"📉 Max drawdown: {stock_data.max_drawdown:.2f}%"
    
    send_telegram_message(message)

# ============================
# EXIT HANDLERS (ENHANCED)
# ============================

def cleanup_and_exit():
    """Enhanced cleanup and exit with detailed summary"""
    print("\n🛑 Enhanced Bulk Trading Bot shutting down...")
    print_final_summary_enhanced()
    
    # Send final telegram summary
    session_duration = datetime.now() - stock_data.session_start
    hours = int(session_duration.total_seconds() // 3600)
    minutes = int((session_duration.total_seconds() % 3600) // 60)
    
    final_message = f"🛑 *Enhanced Bot Stopped*\n"
    final_message += f"⏰ Session: {hours}h {minutes}m\n"
    final_message += f"📊 Total trades: {stock_data.total_trades}\n"
    final_message += f"💰 Final P&L: ₹{stock_data.session_pnl:.2f}\n"
    final_message += f"🏆 Win rate: {(stock_data.profitable_trades/stock_data.total_trades*100):.1f}%" if stock_data.total_trades > 0 else "🏆 Win rate: 0%"
    
    send_telegram_message(final_message)
    sys.exit(0)

def setup_exit_handlers():
    """Setup graceful exit handlers"""
    def signal_handler(sig, frame):
        cleanup_and_exit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(print_final_summary_enhanced)

def print_final_summary_enhanced():
    """Print enhanced final session summary"""
    try:
        session_duration = datetime.now() - stock_data.session_start
        active_positions = sum(1 for pos in stock_data.current_positions.values() if pos.get('shares', 0) > 0)
        
        print("\n" + "="*80)
        print("ENHANCED BULK TRADING SESSION SUMMARY")
        print("="*80)
        print(f"Session Duration: {session_duration}")
        print(f"Stocks Monitored: {len(TICKERS)} with 15+ technical indicators")
        print(f"Total Trades: {stock_data.total_trades}")
        print(f"Profitable Trades: {stock_data.profitable_trades}")
        print(f"Win Rate: {(stock_data.profitable_trades/stock_data.total_trades*100):.1f}%" if stock_data.total_trades > 0 else "Win Rate: 0%")
        print(f"Total P&L: ₹{stock_data.session_pnl:.2f}")
        print(f"Max Drawdown: {stock_data.max_drawdown:.2f}%")
        print(f"Peak Portfolio Value: ₹{stock_data.peak_portfolio_value:.2f}")
        print(f"Active Positions: {active_positions}")
        print(f"Final Market Sentiment: {stock_data.market_sentiment}")
        
        # Show average signal strength
        if stock_data.signal_strengths:
            avg_signal = sum(stock_data.signal_strengths.values()) / len(stock_data.signal_strengths)
            print(f"Average Signal Strength: {avg_signal:.1f}/100")
        
        print("="*80)
        print("Enhanced Features Used:")
        print("• Dynamic position sizing based on volatility & signal strength")
        print("• Advanced trailing stops with ATR and support levels")
        print("• 15+ technical indicators (RSI, MACD, Stochastic, BB, etc.)")
        print("• Smart alerts (breakouts, support levels, 52W highs)")
        print("• Market sentiment analysis")
        print("• Bulk API optimization for 100+ stocks")
        print("="*80)
    except Exception as e:
        print(f"Error in enhanced final summary: {e}")

# ============================
# TIME MANAGEMENT (ENHANCED)
# ============================

def is_market_hours() -> bool:
    """Check if market is open"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    if now.weekday() >= 5:  # Weekend
        return False
    
    return MARKET_START <= current_time <= MARKET_END

def is_alive_check_time() -> bool:
    """Check if it's time for alive notification"""
    current_time = datetime.now().strftime("%H:%M")
    morning_range = "09:15" <= current_time <= "09:30"
    evening_range = "15:00" <= current_time <= "15:15"
    
    return morning_range or evening_range

# ============================
# MAIN ENHANCED TRADING LOOP
# ============================

def main_enhanced_trading_loop():
    """Main enhanced trading loop with advanced features"""
    print("🚀 Enhanced Bulk Stock Trading Bot Started!")
    print(f"📊 Advanced analysis for {len(TICKERS)} stocks with 15+ indicators")
    
    # Send startup message
    startup_msg = f"🚀 *Enhanced Bulk Trading Bot v3.0 Started!*\n"
    startup_msg += f"📊 Monitoring {len(TICKERS)} stocks\n"
    startup_msg += f"⚡ Features: 15+ indicators, Dynamic stops, Smart alerts\n"
    startup_msg += f"🎯 Signal threshold: {STRENGTH_THRESHOLD}+ (Enhanced: 70+)\n"
    startup_msg += f"💾 Memory optimized for bulk processing\n"
    startup_msg += f"📈 Advanced market sentiment analysis enabled"
    
    send_telegram_message(startup_msg)
    
    loop_count = 0
    last_hourly_summary = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            loop_count += 1
            
            # Send alive notifications
            if is_alive_check_time():
                if (stock_data.last_alive_check is None or 
                    (current_time - stock_data.last_alive_check).total_seconds() > 3600):
                    send_enhanced_alive_notification()
            
            # Only trade during market hours
            if not is_market_hours():
                print(f"[{current_time.strftime('%H:%M:%S')}] Market closed. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            print(f"\n[{current_time.strftime('%H:%M:%S')}] Enhanced Analysis Cycle #{loop_count}")
            print(f"🔄 Processing {len(TICKERS)} stocks with advanced indicators...")
            
            # Perform enhanced bulk analysis
            start_time = time.time()
            processed_count = bulk_analyze_stocks_advanced()
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"✅ Enhanced analysis complete: {processed_count} stocks in {processing_time:.1f}s")
            print(f"⚡ Average: {processing_time/len(TICKERS):.2f}s per stock")
            print(f"📈 Market sentiment: {stock_data.market_sentiment}")
            
            # Send hourly detailed summary
            if (current_time - last_hourly_summary).total_seconds() >= 3600:  # Every hour
                send_hourly_summary()
                last_hourly_summary = current_time
            
            # Send periodic summary to Telegram (every 30 minutes)
            if loop_count % 6 == 0:
                active_positions = sum(1 for pos in stock_data.current_positions.values() if pos.get('shares', 0) > 0)
                strong_signals = sum(1 for strength in stock_data.signal_strengths.values() if strength > 75)
                avg_signal = sum(stock_data.signal_strengths.values()) / len(stock_data.signal_strengths) if stock_data.signal_strengths else 0
                
                summary_msg = f"📊 *30min Enhanced Summary*\n"
                summary_msg += f"💼 Active: {active_positions}/{len(TICKERS)} positions\n"
                summary_msg += f"🎯 Strong signals: {strong_signals} (Avg: {avg_signal:.1f})\n"
                summary_msg += f"📈 Market: {stock_data.market_sentiment}\n"
                summary_msg += f"💰 P&L: ₹{stock_data.session_pnl:.2f}\n"
                summary_msg += f"🚀 Processing: {processing_time:.1f}s for {len(TICKERS)} stocks\n"
                summary_msg += f"🏆 Win rate: {(stock_data.profitable_trades/stock_data.total_trades*100):.1f}%" if stock_data.total_trades > 0 else "🏆 Win rate: 0%"
                
                send_telegram_message(summary_msg)
            
            print(f"[{current_time.strftime('%H:%M:%S')}] Next enhanced analysis in {CHECK_INTERVAL//60} minutes...")
            
            # Force garbage collection after each cycle
            gc.collect()
            
        except KeyboardInterrupt:
            print("\n🛑 Enhanced Bot stopped by user")
            cleanup_and_exit()
            break
        except Exception as e:
            print(f"Error in main enhanced loop: {e}")
            error_msg = f"❌ *Enhanced Bot Error*\nCycle #{loop_count}\nError: {str(e)[:100]}\nBot continuing with advanced features..."
            send_telegram_message(error_msg)
        
        time.sleep(CHECK_INTERVAL)

# ============================
# ENHANCED HEALTH CHECK
# ============================

def perform_enhanced_health_check():
    """Perform enhanced system health check"""
    print("\n🏥 Enhanced System Health Check:")
    print(f"  📊 Tickers configured: {len(TICKERS)}")
    print(f"  📦 Bulk fetch size: {BULK_FETCH_SIZE}")
    print(f"  ⏰ API delay: {API_DELAY}s")
    print(f"  🔄 Check interval: {CHECK_INTERVAL//60} minutes")
    print(f"  🎯 Enhanced signal threshold: 70+ (vs basic 65+)")
    print(f"  📈 ATR multiplier: {ATR_MULTIPLIER}")
    print(f"  📊 Volume spike threshold: {MIN_VOLUME_SPIKE}x")
    
    # Test API connectivity with multiple stocks
    try:
        print("  🧪 Testing bulk API with sample stocks...")
        test_tickers = TICKERS[:3]
        test_data = bulk_fetch_stock_data(test_tickers, period="1mo")
        
        if len(test_data) >= len(test_tickers) * 0.8:  # At least 80% success
            print(f"  ✅ Bulk API test: EXCELLENT ({len(test_data)}/{len(test_tickers)} stocks)")
        elif len(test_data) >= len(test_tickers) * 0.5:  # At least 50% success
            print(f"  ⚠️ Bulk API test: FAIR ({len(test_data)}/{len(test_tickers)} stocks)")
        else:
            print(f"  ❌ Bulk API test: POOR ({len(test_data)}/{len(test_tickers)} stocks)")
        
        # Test advanced indicators
        if test_data:
            sample_ticker = list(test_data.keys())[0]
            sample_df = test_data[sample_ticker]
            indicators = calculate_advanced_indicators(sample_df)
            indicator_count = len([v for v in indicators.values() if v is not None])
            print(f"  ✅ Advanced indicators: {indicator_count} calculated successfully")
        
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
    
    # Check Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'YOUR_BOT_TOKEN_HERE':
        print(f"  ✅ Telegram: Configured for {len(TELEGRAM_CHAT_ID)} chats")
    else:
        print(f"  ⚠️ Telegram: Not configured (console mode)")
    
    # Memory info
    try:
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"  💾 Memory usage: {memory_mb:.1f} MB")
        if memory_mb > 400:
            print(f"  ⚠️ High memory usage - consider reducing BULK_FETCH_SIZE")
    except ImportError:
        print(f"  💾 Memory monitoring: psutil not available")
    
    print("🏥 Enhanced health check complete\n")

# ============================
# ENTRY POINT
# ============================



# ============================
# BULK DATA FETCHING FUNCTIONS (ENHANCED)
# ============================

def bulk_fetch_stock_data(ticker_list: List[str], period: str = "6mo") -> Dict[str, pd.DataFrame]:
    """Fetch stock data in bulk with retry mechanism"""
    stock_data_dict = {}
    
    # Split tickers into chunks
    ticker_chunks = [ticker_list[i:i + BULK_FETCH_SIZE] for i in range(0, len(ticker_list), BULK_FETCH_SIZE)]
    
    for chunk_idx, ticker_chunk in enumerate(ticker_chunks):
        print(f"Fetching bulk data chunk {chunk_idx + 1}/{len(ticker_chunks)} ({len(ticker_chunk)} stocks)...")
        
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # Create ticker string for bulk download
                ticker_string = ' '.join(ticker_chunk)
                
                # Bulk download using yfinance
                bulk_data = yf.download(
                    tickers=ticker_string,
                    period=period,
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=True,
                    threads=True,
                    progress=False,
                    timeout=REQUEST_TIMEOUT
                )
                
                if bulk_data.empty:
                    print(f"Warning: No data received for chunk {chunk_idx + 1}")
                    break
                
                # Process each ticker from bulk data
                for ticker in ticker_chunk:
                    try:
                        if len(ticker_chunk) == 1:
                            # Single ticker case
                            ticker_data = bulk_data
                        else:
                            # Multi-ticker case
                            ticker_data = bulk_data[ticker] if ticker in bulk_data.columns.get_level_values(0) else None
                        
                        if ticker_data is not None and not ticker_data.empty:
                            # Clean the data and add enhanced features
                            ticker_data = ticker_data.dropna()
                            if len(ticker_data) > 0:
                                # Add additional calculated columns
                                ticker_data['Returns'] = ticker_data['Close'].pct_change()
                                ticker_data['Volatility'] = ticker_data['Returns'].rolling(window=20).std() * np.sqrt(252)
                                
                                stock_data_dict[ticker] = ticker_data
                                print(f"  ✓ {ticker}: {len(ticker_data)} records")
                            else:
                                print(f"  ⚠ {ticker}: No valid data")
                        else:
                            print(f"  ✗ {ticker}: Failed to fetch")
                            
                    except Exception as e:
                        print(f"  ✗ {ticker}: Error processing - {e}")
                        continue
                
                print(f"Chunk {chunk_idx + 1} completed. Waiting {API_DELAY}s...")
                time.sleep(API_DELAY)
                break

            except requests.exceptions.Timeout:
                print(f"  Timeout for chunk {chunk_idx + 1}, retrying...")
                retry_count += 1
                time.sleep(API_DELAY * 2)
                continue
            except requests.exceptions.ConnectionError:
                print(f"  Connection error for chunk {chunk_idx + 1}, retrying...")
                retry_count += 1
                time.sleep(API_DELAY * 3)
                continue    
            except Exception as e:
                retry_count += 1
                print(f"  Retry {retry_count}/{MAX_RETRIES} for chunk {chunk_idx + 1}: {e}")
                if retry_count < MAX_RETRIES:
                    time.sleep(API_DELAY * retry_count)
                else:
                    print(f"  Failed to fetch chunk {chunk_idx + 1} after {MAX_RETRIES} retries")
    
    print(f"Bulk fetch complete: {len(stock_data_dict)}/{len(ticker_list)} stocks successfully fetched")
    return stock_data_dict

def bulk_fetch_realtime_data(ticker_list: List[str]) -> Dict[str, Dict]:
    """Fetch real-time data in bulk with price history tracking"""
    realtime_data_dict = {}
    
    # Split tickers into chunks for real-time data
    ticker_chunks = [ticker_list[i:i + BULK_FETCH_SIZE] for i in range(0, len(ticker_list), BULK_FETCH_SIZE)]
    
    for chunk_idx, ticker_chunk in enumerate(ticker_chunks):
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # Create ticker string for bulk download
                ticker_string = ' '.join(ticker_chunk)
                
                # Fetch recent data (2 days, 5-minute intervals)
                bulk_data = yf.download(
                    tickers=ticker_string,
                    period="2d",
                    interval="5m",
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=True,
                    threads=True,
                    progress=False,
                    timeout=REQUEST_TIMEOUT
                )
                
                if bulk_data.empty:
                    print(f"Warning: No real-time data for chunk {chunk_idx + 1}")
                    break
                
                # Process each ticker from bulk data
                for ticker in ticker_chunk:
                    try:
                        if len(ticker_chunk) == 1:
                            ticker_data = bulk_data
                        else:
                            ticker_data = bulk_data[ticker] if ticker in bulk_data.columns.get_level_values(0) else None
                        
                        if ticker_data is not None and not ticker_data.empty:
                            ticker_data = ticker_data.dropna()
                            if len(ticker_data) > 0:
                                current_price = float(ticker_data['Close'].iloc[-1])
                                current_volume = float(ticker_data['Volume'].iloc[-1])
                                day_high = float(ticker_data['High'].max())
                                day_low = float(ticker_data['Low'].min())
                                day_open = float(ticker_data['Open'].iloc[0])
                                day_change = ((current_price - day_open) / day_open) * 100 if day_open > 0 else 0
                                
                                # Update price and volume history
                                stock_data.price_history[ticker].append(current_price)
                                stock_data.volume_history[ticker].append(current_volume)
                                
                                realtime_data_dict[ticker] = {
                                    'price': current_price,
                                    'volume': current_volume,
                                    'high': float(ticker_data['High'].iloc[-1]),
                                    'low': float(ticker_data['Low'].iloc[-1]),
                                    'day_high': day_high,
                                    'day_low': day_low,
                                    'day_open': day_open,
                                    'day_change': day_change
                                }
                    
                    except Exception as e:
                        print(f"  Error processing real-time data for {ticker}: {e}")
                        continue
                
                time.sleep(API_DELAY / 2)  # Shorter delay for real-time data
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    time.sleep(API_DELAY * retry_count / 2)
                else:
                    print(f"Failed to fetch real-time data for chunk {chunk_idx + 1}")
    
    return realtime_data_dict

# ============================
# ADVANCED TECHNICAL INDICATORS
# ============================

def calculate_advanced_indicators(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive technical indicators (15+ indicators)"""
    try:
        if len(df) < 50:
            print("Not enough data for advanced indicators calculation")
            return {}
            
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        volume = df['Volume'].values
        
        indicators = {}
        
        # === TREND INDICATORS ===
        # Multiple SMAs for trend analysis
        indicators['sma_9'] = talib.SMA(close_prices, timeperiod=9)
        indicators['sma_20'] = talib.SMA(close_prices, timeperiod=20)
        indicators['sma_50'] = talib.SMA(close_prices, timeperiod=50)
        indicators['sma_200'] = talib.SMA(close_prices, timeperiod=200)
        
        # Exponential Moving Averages
        indicators['ema_12'] = talib.EMA(close_prices, timeperiod=12)
        indicators['ema_26'] = talib.EMA(close_prices, timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        
        # === MOMENTUM INDICATORS ===
        # RSI with different periods
        indicators['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
        indicators['rsi_21'] = talib.RSI(close_prices, timeperiod=21)
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        # Williams %R
        indicators['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # === VOLATILITY INDICATORS ===
        # ATR with multiple periods
        indicators['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        indicators['atr_21'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=21)
        
        # Bollinger Bands
        # bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=1.8, nbdevdn=1.8)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # === VOLUME INDICATORS ===
        # Volume SMAs
        indicators['volume_sma_10'] = talib.SMA(volume.astype(float), timeperiod=10)
        indicators['volume_sma_30'] = talib.SMA(volume.astype(float), timeperiod=30)
        
        # On Balance Volume
        indicators['obv'] = talib.OBV(close_prices, volume.astype(float))
        
        # Volume Rate of Change
        indicators['volume_roc'] = talib.ROC(volume.astype(float), timeperiod=10)
        
        # === SUPPORT/RESISTANCE ===
        # Pivot Points
        indicators['pivot_point'] = (high_prices[-1] + low_prices[-1] + close_prices[-1]) / 3
        indicators['resistance_1'] = 2 * indicators['pivot_point'] - low_prices[-1]
        indicators['support_1'] = 2 * indicators['pivot_point'] - high_prices[-1]
        
        # === CUSTOM INDICATORS ===
        # Price momentum
        if len(close_prices) >= 5:
            indicators['momentum_5'] = (close_prices[-1] - close_prices[-5]) / close_prices[-5] * 100
        
        # Volatility ratio
        if len(close_prices) >= 20:
            recent_volatility = np.std(close_prices[-10:]) / np.mean(close_prices[-10:])
            historical_volatility = np.std(close_prices[-20:-10]) / np.mean(close_prices[-20:-10])
            indicators['volatility_ratio'] = recent_volatility / historical_volatility if historical_volatility > 0 else 1
        
        # 52-week high/low
        indicators['52w_high'] = float(df['High'].max())
        indicators['52w_low'] = float(df['Low'].min())
        indicators['distance_from_52w_high'] = ((indicators['52w_high'] - close_prices[-1]) / indicators['52w_high']) * 100
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating advanced indicators: {e}")
        return {}

def calculate_market_sentiment() -> str:
    """Calculate overall market sentiment based on all stocks"""
    try:
        bullish_count = 0
        bearish_count = 0
        total_strength = 0
        
        for ticker in TICKERS:
            strength = stock_data.signal_strengths.get(ticker, 0)
            total_strength += strength
            
            if strength > 70:
                bullish_count += 1
            elif strength < 40:
                bearish_count += 1
        
        avg_strength = total_strength / len(TICKERS) if TICKERS else 50
        bullish_ratio = bullish_count / len(TICKERS)
        bearish_ratio = bearish_count / len(TICKERS)
        
        if bullish_ratio >= 0.6 or avg_strength > 70:
            return 'BULLISH'
        elif bearish_ratio >= 0.6 or avg_strength < 35:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
            
    except Exception as e:
        print(f"Error calculating market sentiment: {e}")
        return 'NEUTRAL'

def advanced_should_sell(ticker: str, indicators: Dict, current_price: float) -> Tuple[bool, str]:
    """Advanced sell signal with dynamic conditions"""
    if ticker not in stock_data.current_positions or stock_data.current_positions[ticker].get('shares', 0) == 0:
        return False, "No position"
    
    try:
        entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
        current_pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # Dynamic stop-loss check
        if ticker in stock_data.stop_losses and current_price <= stock_data.stop_losses[ticker]:
            return True, f"Dynamic stop-loss (PnL: {current_pnl:+.2f}%)"
        
        # Advanced profit-taking conditions
        if current_pnl > 10:  # 10% profit threshold
            sell_conditions = []
            
            # RSI overbought
            rsi = safe_extract(indicators.get('rsi_14'))
            if rsi and rsi > 75:
                sell_conditions.append("RSI overbought")
            
            # Stochastic overbought
            stoch_k = safe_extract(indicators.get('stoch_k'))
            if stoch_k and stoch_k > 75: # 80:
                sell_conditions.append("Stoch overbought")
            
            # MACD divergence
            macd = safe_extract(indicators.get('macd'))
            macd_signal = safe_extract(indicators.get('macd_signal'))
            if macd and macd_signal and macd < macd_signal:
                sell_conditions.append("MACD bearish")
            
            if len(sell_conditions) >= 2:
                return True, f"Profit-take (PnL: {current_pnl:+.2f}%): " + ", ".join(sell_conditions[:2])
        
        # Trend reversal detection
        sma_20 = safe_extract(indicators.get('sma_20'))
        ema_12 = safe_extract(indicators.get('ema_12'))
        ema_26 = safe_extract(indicators.get('ema_26'))
        
        if all([sma_20, ema_12, ema_26]):
            if current_price < sma_20 * 0.98 and ema_12 < ema_26:
                return True, f"Trend reversal (PnL: {current_pnl:+.2f}%)"
        
        # Bollinger Band squeeze exit
        bb_upper = safe_extract(indicators.get('bb_upper'))
        bb_lower = safe_extract(indicators.get('bb_lower'))
        if bb_upper and bb_lower and current_price > bb_upper:
            volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
            if volatility_ratio and volatility_ratio > 2.5:  # High volatility
                return True, f"Volatility exit (PnL: {current_pnl:+.2f}%)"
        
        # Time-based exit with profit conditions
        if ticker in stock_data.current_positions and 'entry_time' in stock_data.current_positions[ticker]:
            holding_time = datetime.now() - stock_data.current_positions[ticker]['entry_time']
            if holding_time.days > 5 and current_pnl < 2:  # 7 days with minimal profit
                return True, f"Time exit (PnL: {current_pnl:+.2f}%, {holding_time.days}d)"
        
        return False, f"Hold (PnL: {current_pnl:+.2f}%)"
        
    except Exception as e:
        print(f"Error in advanced_should_sell for {ticker}: {e}")
        return False, "Error in sell analysis"

def execute_advanced_buy(ticker: str, current_price: float, indicators: Dict, reason: str):
    """Execute advanced buy with dynamic position sizing"""
    try:
        atr = safe_extract(indicators.get('atr_14'))
        if atr is None or atr <= 0:
            atr = current_price * 0.02
        
        # Dynamic position sizing based on volatility and signal strength
        volatility_ratio = safe_extract(indicators.get('volatility_ratio'), 1.0)
        signal_strength = stock_data.signal_strengths.get(ticker, 0)
        
        # Adjust shares based on signal strength and volatility
        base_shares = SHARES_TO_BUY
        if signal_strength > 85:
            base_shares = int(SHARES_TO_BUY * 1.5)  # Increase position for very strong signals
        elif signal_strength > 75:
            base_shares = int(SHARES_TO_BUY * 1.2)
        
        adjusted_shares = max(1, int(base_shares / volatility_ratio))
        
        stock_data.current_positions[ticker] = {
            'shares': adjusted_shares,
            'entry_price': current_price,
            'entry_time': datetime.now()
        }
        
        # Dynamic stop-loss calculation
        support_level = safe_extract(indicators.get('support_1'), current_price * 0.94)
        atr_stop = current_price - (ATR_MULTIPLIER * atr)
        bb_lower = safe_extract(indicators.get('bb_lower'))
        
        # Use the highest of support, ATR stop, or BB lower band
        dynamic_stops = [support_level, atr_stop]
        if bb_lower:
            dynamic_stops.append(bb_lower * 0.98)
        
        dynamic_stop = max(dynamic_stops)
        stock_data.stop_losses[ticker] = dynamic_stop
        stock_data.highest_prices[ticker] = current_price
        
        stock_data.total_trades += 1
        
        # Enhanced buy notification
        symbol = ticker.replace('.NS', '')
        rsi_val = safe_extract(indicators.get('rsi_14'))
        signal_strength = stock_data.signal_strengths.get(ticker, 0)
        
        message = f"🟢 *ADVANCED BUY*\n"
        message += f"📈 {symbol} @ ₹{current_price:.2f}\n"
        message += f"💰 Qty: {adjusted_shares} (Dynamic)\n"
        message += f"🎯 Signal: {signal_strength:.1f}/100\n"
        message += f"📊 RSI: {rsi_val:.1f} | ATR: ₹{atr:.2f}\n"
        message += f"🛑 Smart Stop: ₹{dynamic_stop:.2f}\n"
        message += f"💡 {reason}\n"
        message += f"📈 Market: {stock_data.market_sentiment}"
        
        send_telegram_message(message)
        print(f"[ADVANCED BUY] {symbol} @ ₹{current_price:.2f} | Qty: {adjusted_shares} | Signal: {signal_strength:.1f}")
        
    except Exception as e:
        print(f"Buy execution error for {ticker}: {e}")

def execute_advanced_sell(ticker: str, current_price: float, reason: str):
    """Execute advanced sell with detailed tracking"""
    try:
        if ticker not in stock_data.current_positions:
            return
        
        shares = stock_data.current_positions[ticker].get('shares', 0)
        entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
        entry_time = stock_data.current_positions[ticker].get('entry_time', datetime.now())
        
        if shares == 0:
            return
        
        # Calculate detailed P&L
        total_change = (current_price - entry_price) * shares
        change_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        holding_period = datetime.now() - entry_time
        
        # Update session statistics
        stock_data.session_pnl += total_change
        if total_change > 0:
            stock_data.profitable_trades += 1
        
        # Update drawdown tracking
        current_portfolio_value = stock_data.session_pnl
        if current_portfolio_value > stock_data.peak_portfolio_value:
            stock_data.peak_portfolio_value = current_portfolio_value
        else:
            drawdown = ((stock_data.peak_portfolio_value - current_portfolio_value) / stock_data.peak_portfolio_value) * 100
            if drawdown > stock_data.max_drawdown:
                stock_data.max_drawdown = drawdown
        
        # Clear position and reset alerts
        stock_data.current_positions[ticker] = {'shares': 0, 'entry_price': 0}
        if ticker in stock_data.stop_losses:
            del stock_data.stop_losses[ticker]
        if ticker in stock_data.highest_prices:
            del stock_data.highest_prices[ticker]
        
        # Reset alerts
        stock_data.alerts_sent[ticker] = {'52w_high': False, 'breakout': False, 'support': False}
        
        symbol = ticker.replace('.NS', '')
        profit_emoji = "💚" if total_change >= 0 else "❌"
        holding_days = holding_period.days
        holding_hours = holding_period.seconds // 3600
        
        message = f"🔴 *ADVANCED SELL*\n"
        message += f"📉 {symbol} @ ₹{current_price:.2f}\n"
        message += f"💼 Qty: {shares}\n"
        message += f"{profit_emoji} P&L: ₹{total_change:.2f} ({change_percent:+.2f}%)\n"
        message += f"⏱️ Held: {holding_days}d {holding_hours}h\n"
        message += f"💡 {reason}\n"
        message += f"📊 Session P&L: ₹{stock_data.session_pnl:.2f}"
        
        send_telegram_message(message)
        print(f"[ADVANCED SELL] {symbol} @ ₹{current_price:.2f} | P&L: ₹{total_change:.2f}")
        
    except Exception as e:
        print(f"Sell execution error for {ticker}: {e}")

def update_dynamic_trailing_stop(ticker: str, current_price: float, indicators: Dict):
    """Update dynamic trailing stop-loss with ATR and support levels"""
    if ticker not in stock_data.current_positions or stock_data.current_positions[ticker].get('shares', 0) == 0:
        return
    
    try:
        atr = safe_extract(indicators.get('atr_14'))
        if atr is None or atr <= 0:
            atr = current_price * 0.02
        
        # Update highest price
        if ticker not in stock_data.highest_prices:
            stock_data.highest_prices[ticker] = current_price
        else:
            stock_data.highest_prices[ticker] = max(stock_data.highest_prices[ticker], current_price)
        
        # Dynamic ATR multiplier based on profit
        entry_price = stock_data.current_positions[ticker].get('entry_price', current_price)
        profit_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # Tighten stop-loss as profit increases
        if profit_percent > 15:
            atr_multiplier = 1.2  # Very tight stop for high profits
        elif profit_percent > 10:
            atr_multiplier = 1.5  # Tight stop
        elif profit_percent > 5:
            atr_multiplier = 1.8  # Medium stop
        else:
            atr_multiplier = ATR_MULTIPLIER  # Standard stop
        
        # Calculate new trailing stop
        new_stop = stock_data.highest_prices[ticker] - (atr_multiplier * atr)
        
        # Support level consideration
        support_level = safe_extract(indicators.get('support_1'))
        if support_level and support_level < new_stop:
            new_stop = max(new_stop, support_level * 0.98)  # Small buffer below support
        
        # Bollinger lower band consideration
        bb_lower = safe_extract(indicators.get('bb_lower'))
        if bb_lower and bb_lower > entry_price:  # Only if BB lower is above entry
            new_stop = max(new_stop, bb_lower * 0.98)
        
        # Only update if new stop is higher (trailing up)
        if ticker not in stock_data.stop_losses:
            stock_data.stop_losses[ticker] = new_stop
        else:
            stock_data.stop_losses[ticker] = max(stock_data.stop_losses[ticker], new_stop)
            
    except Exception as e:
        print(f"Error updating trailing stop for {ticker}: {e}")

def check_advanced_alerts(ticker: str, current_price: float, indicators: Dict, realtime_data: Dict):
    """Check for advanced alerts and notifications"""
    try:
        symbol = ticker.replace('.NS', '')
        
        # 1. Breakout Alert
        if not stock_data.alerts_sent[ticker]['breakout']:
            bb_upper = safe_extract(indicators.get('bb_upper'))
            volume_sma = safe_extract(indicators.get('volume_sma_10'))
            current_volume = realtime_data.get('volume', 0)
            rsi = safe_extract(indicators.get('rsi_14'))
            
            # Enhanced breakout conditions
            if all([bb_upper, volume_sma, rsi]) and current_price > bb_upper:
                volume_spike = current_volume > volume_sma * 2 if volume_sma > 0 else False
                rsi_condition = 50 < rsi < 80  # Not overbought breakout
                
                if volume_spike and rsi_condition:
                    message = f"🚀 *BREAKOUT ALERT*\n"
                    message += f"📈 {symbol} - Bullish breakout!\n"
                    message += f"💰 Price: ₹{current_price:.2f} vs BB: ₹{bb_upper:.2f}\n"
                    message += f"📊 Volume: {(current_volume/volume_sma):.1f}x avg\n"
                    message += f"🎯 RSI: {rsi:.1f} (Healthy breakout)\n"
                    message += f"⚡ Consider position entry"
                    
                    send_telegram_message(message)
                    stock_data.alerts_sent[ticker]['breakout'] = True
        
        # 2. Support Level Alert (for holdings)
        if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
            if not stock_data.alerts_sent[ticker]['support']:
                support_level = safe_extract(indicators.get('support_1'))
                bb_lower = safe_extract(indicators.get('bb_lower'))
                
                support_alerts = []
                if support_level and current_price < support_level * 1.03:
                    support_alerts.append(f"Key support: ₹{support_level:.2f}")
                
                if bb_lower and current_price < bb_lower * 1.02:
                    support_alerts.append(f"BB lower: ₹{bb_lower:.2f}")
                
                if support_alerts:
                    entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
                    pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    
                    message = f"⚠️ *SUPPORT ALERT*\n"
                    message += f"📉 {symbol} near support levels\n"
                    message += f"💰 Current: ₹{current_price:.2f}\n"
                    message += f"🛡️ {' | '.join(support_alerts)}\n"
                    message += f"📊 Your P&L: {pnl:+.2f}%\n"
                    message += f"👀 Watch for bounce or exit"
                    
                    send_telegram_message(message)
                    stock_data.alerts_sent[ticker]['support'] = True
        
        # 3. 52-Week High Alert (for holdings)
        if ticker in stock_data.current_positions and stock_data.current_positions[ticker].get('shares', 0) > 0:
            if not stock_data.alerts_sent[ticker]['52w_high']:
                high_52w = indicators.get('52w_high', 0)
                distance_from_high = safe_extract(indicators.get('distance_from_52w_high'))
                
                if distance_from_high and distance_from_high < 3:  # Within 3% of 52w high
                    entry_price = stock_data.current_positions[ticker].get('entry_price', 0)
                    profit = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    rsi = safe_extract(indicators.get('rsi_14'))
                    
                    message = f"🏆 *52W HIGH ALERT*\n"
                    message += f"🚀 {symbol} near yearly peak!\n"
                    message += f"📊 Current: ₹{current_price:.2f}\n"
                    message += f"🎯 52W High: ₹{high_52w:.2f}\n"
                    message += f"💚 Your gain: {profit:+.2f}%\n"
                    if rsi:
                        message += f"📈 RSI: {rsi:.1f}\n"
                    message += f"🤔 Consider profit booking"
                    
                    send_telegram_message(message)
                    stock_data.alerts_sent[ticker]['52w_high'] = True
        
        # 4. Volume Surge Alert (for watchlist)
        volume_sma_30 = safe_extract(indicators.get('volume_sma_30'))
        current_volume = realtime_data.get('volume', 0)
        if volume_sma_30 and current_volume > volume_sma_30 * 3:  # 3x volume surge
            # Only send if not already in position and signal is strong
            if (ticker not in stock_data.current_positions or 
                stock_data.current_positions[ticker].get('shares', 0) == 0):
                signal_strength = stock_data.signal_strengths.get(ticker, 0)
                if signal_strength > 60:
                    day_change = realtime_data.get('day_change', 0)
                    message = f"📢 *VOLUME SURGE*\n"
                    message += f"⚡ {symbol} - Unusual activity\n"
                    message += f"💰 Price: ₹{current_price:.2f} ({day_change:+.2f}%)\n"
                    message += f"📊 Volume: {(current_volume/volume_sma_30):.1f}x normal\n"
                    message += f"🎯 Signal: {signal_strength:.0f}/100\n"
                    message += f"👁️ Worth monitoring"
                    
                    # Don't spam - limit to strong signals only
                    if signal_strength > 70:
                        send_telegram_message(message)
                    
    except Exception as e:
        print(f"Error checking alerts for {ticker}: {e}")

def has_earnings_soon(ticker: str) -> bool:
    """Check if stock has earnings in next 2 days (optional filter)"""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is not None and not calendar.empty:
            next_earnings = pd.to_datetime(calendar.index[0])
            days_until = (next_earnings - datetime.now()).days
            return days_until <= 2
    except:
        pass
    return False

if __name__ == "__main__":
    # Set up exit handlers first
    setup_exit_handlers()
    
    # Verify required libraries
    try:
        import talib
        from tabulate import tabulate
        print("✅ All required libraries verified")
    except ImportError as e:
        if 'talib' in str(e):
            print("ERROR: TA-Lib not installed. Install with: pip install TA-Lib")
            print("On Windows, you might need to download the wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        elif 'tabulate' in str(e):
            print("ERROR: tabulate not installed. Install with: pip install tabulate")
        sys.exit(1)
    
    # Configuration check
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("⚠️ WARNING: Telegram bot token not configured. Messages will print to console.")
    
    print("🔧 Enhanced Bulk Trading Configuration:")
    print(f"   📊 Total stocks: {len(TICKERS)}")
    print(f"   📦 Bulk fetch size: {BULK_FETCH_SIZE}")
    print(f"   ⏰ Check interval: {CHECK_INTERVAL//60} minutes")
    print(f"   🚀 API delay: {API_DELAY} seconds")
    print(f"   🔄 Max retries: {MAX_RETRIES}")
    print(f"   📈 ATR Multiplier: {ATR_MULTIPLIER}")
    print(f"   🎯 Enhanced signal threshold: 70+ (vs basic 65)")
    print(f"   📊 Volume spike threshold: {MIN_VOLUME_SPIKE}x")
    print(f"   💾 Memory optimized: YES")
    print(f"   🚀 Advanced features: 15+ indicators, Dynamic stops, Smart alerts")
    
    # Perform enhanced health check
    perform_enhanced_health_check()
    
    # Initial system test with advanced features
    print("🔍 Testing enhanced bulk API with first 3 stocks...")
    try:
        test_tickers = TICKERS[:3]
        test_historical = bulk_fetch_stock_data(test_tickers, period="1mo")
        test_realtime = bulk_fetch_realtime_data(test_tickers)
        
        success_rate = len(test_historical) / len(test_tickers) * 100
        print(f"✅ Enhanced API test: {success_rate:.1f}% success rate")
        
        # Test advanced indicators
        if test_historical:
            sample_ticker = list(test_historical.keys())[0]
            sample_df = test_historical[sample_ticker]
            indicators = calculate_advanced_indicators(sample_df)
            print(f"✅ Advanced indicators test: {len(indicators)} indicators calculated")
            
            # Test signal strength calculation
            if test_realtime and sample_ticker in test_realtime:
                rt_data = test_realtime[sample_ticker]
                signal = calculate_advanced_signal_strength(sample_ticker, indicators, rt_data['price'], rt_data)
                print(f"✅ Signal strength test: {signal:.1f}/100")
        
    except Exception as e:
        print(f"❌ Enhanced API test failed: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🚀 ENHANCED BULK TRADING BOT - READY TO START")
    print("Features: 15+ Technical Indicators | Dynamic Position Sizing")
    print("Smart Alerts | Advanced Risk Management | Market Sentiment Analysis")
    print("="*80)
    
    # Start the enhanced trading bot
    try:
        main_enhanced_trading_loop()
    except Exception as e:
        print(f"Fatal error: {e}")
        cleanup_and_exit()
    finally:
        print_final_summary_enhanced() 