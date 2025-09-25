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
import os
import logging
import psutil
import subprocess
# from enhanced_diagnostics import setup_enhanced_diagnostics, monitor_system_health

# import json  



warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Suppress noisy external library logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("yfinance").disabled = True


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler('advanced_trading_bot.log', mode='a', encoding='utf-8')  # File output
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)
# ============================
# CONFIGURATION
# ============================

# Telegram Configuration
# TELEGRAM_BOT_TOKEN = '7933607173:AAFND1Z_GxNdvKwOc4Y_LUuX327eEpc2KIE'
# TELEGRAM_CHAT_ID = ['1012793457','1209666577']

# # Trading Configuration
# TICKERS = [
# "INFY.NS","ITC.NS","TATAMOTORS.NS","DABUR.NS","ACC.NS","TATACONSUM.NS",
# "BAJFINANCE.NS","MUTHOOTFIN.NS","IDFCFIRSTB.NS","RBLBANK.NS","AUBANK.NS",
# "ADANIPORTS.NS","COALINDIA.NS","BPCL.NS","TATAPOWER.NS","RVNL.NS","BEL.NS","LAURUSLABS.NS","NEULANDLAB.NS",
# "VEDL.NS","SAIL.NS","NMDC.NS","NATIONALUM.NS",
# "COCHINSHIP.NS","GRSE.NS","RITES.NS",
# "AFFLE.NS","KSOLVES.NS","NYKAA.NS",
# "GSFC.NS","GPIL.NS","CENTEXT.NS",
# "OLECTRA.NS","BDL.NS","MANAPPURAM.NS","INDRAMEDCO.NS"
# ]
# [
# "BAJFINANCE.NS","BAJAJFINSV.NS","ADANIPORTS.NS","COALINDIA.NS","DRREDDY.NS","BPCL.NS","VEDL.NS","SAIL.NS","NMDC.NS","MOIL.NS",
# "TATACONSUM.NS","MUTHOOTFIN.NS","MANAPPURAM.NS","LICHSGFIN.NS","UNIONBANK.NS","AUBANK.NS","IDBI.NS","IDEA.NS","RPOWER.NS","TATAPOWER.NS",
# "FILATFASH.NS","SRESTHA.BO","JAMNAAUTO.NS","OLECTRA.NS","AFFLE.NS","BEL.NS","COCHINSHIP.NS","RVNL.NS","BDL.NS","NATIONALUM.NS",
# "ARIHANTCAP.NS","GEOJITFSL.NS","MCLOUD.NS","CENTEXT.NS","RNBDENIMS.BO","GPIL.NS","TALBROAUTO.NS","GALLANTT.NS","RESPONIND.NS","NAM-INDIA.NS",
# "ELECON.NS","GRSE.NS","RITES.NS","ZENTEC.NS","NEULANDLAB.NS","EMIL.NS","ACC.NS","ADANIPOWER.NS","ARE&M.NS","BTML.BO",
# "COLPAL.NS","DABUR.NS","GSFC.NS","IDFCFIRSTB.NS","INDRAMEDCO.NS","INFY.NS","ITC.NS","KSOLVES.NS","LAURUSLABS.NS","NESTLEIND.NS",
# "NYKAA.NS","RBLBANK.NS","TATAMOTORS.NS"
# ]

# [
#   "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ITC.NS","BHARTIARTL.NS","SBIN.NS","LT.NS","HCLTECH.NS","WIPRO.NS",
#   "HINDUNILVR.NS","BAJFINANCE.NS","MARUTI.NS","KOTAKBANK.NS","ASIANPAINT.NS","NESTLEIND.NS","DMART.NS","BAJAJFINSV.NS","TITAN.NS","ADANIPORTS.NS",
#   "AXISBANK.NS","ICICIBANK.NS","SUNPHARMA.NS","ULTRACEMCO.NS","NTPC.NS","POWERGRID.NS","TECHM.NS","M&M.NS","TATAMOTORS.NS","COALINDIA.NS",
#   "JSWSTEEL.NS","TATASTEEL.NS","INDUSINDBK.NS","GRASIM.NS","DRREDDY.NS","BRITANNIA.NS","CIPLA.NS","DIVISLAB.NS","EICHERMOT.NS","HEROMOTOCO.NS",
#   "HINDALCO.NS","BAJAJ-AUTO.NS","SHREE.NS","APOLLOHOSP.NS","ONGC.NS","IOC.NS","BPCL.NS","HDFCLIFE.NS","SBILIFE.NS","VEDL.NS",
#   "ADANIGREEN.NS","ADANIENT.NS","ADANITRANS.NS","GODREJCP.NS","PIDILITIND.NS","DABUR.NS","MARICO.NS","COLPAL.NS","MCDOWELL-N.NS","BERGEPAINT.NS",
#   "AMBUJACEM.NS","ACC.NS","SHREECEM.NS","RAMCO.NS","DALBHARAT.NS","SAIL.NS","NMDC.NS","MOIL.NS","GMRINFRA.NS","IRCTC.NS",
#   "ZOMATO.NS","NYKAA.NS","PAYTM.NS","POLICYBZR.NS","TATACONSUM.NS","MUTHOOTFIN.NS","BAJAJHLDNG.NS","CHOLAFIN.NS","LTF.NS","MANAPPURAM.NS",
#   "RECLTD.NS","PFC.NS","LICHSGFIN.NS","HUDCO.NS","CANBK.NS","PNB.NS","BANKBARODA.NS","UNIONBANK.NS","IDFCFIRSTB.NS","FEDERALBNK.NS",
#   "RBLBANK.NS","BANDHANBNK.NS","AUBANK.NS","YESBANK.NS","IDBI.NS","IDEA.NS","BHARTI.NS","RCOM.NS","MTNL.NS","BSNL.NS",
#   "RELCAPITAL.NS","RPOWER.NS","ADANIPOWER.NS","TATAPOWER.NS","NHPC.NS","FILATFASH.NS","SRESTHA.BO","HARSHILAGR.BO","GTLINFRA.NS","OBEROIRLTY.NS",
#   "JAMNAAUTO.NS","KSOLVES.NS","OLECTRA.NS","ARE&M.NS","AFFLE.NS","BEL.NS","LAURUSLABS.NS","KRBL.NS","GSFC.NS","SVARTCORP.BO",
#   "SWASTIVI.BO","BTML.NS","SULABEN.BO","CRYSTAL.BO","TILAK.BO","COMFINTE.BO","COCHINSHIP.NS","RVNL.NS","SHAILY.NS","BDL.NS",
#   "JYOTICNC.NS","NATIONALUM.NS","KRONOX.NS","SAKSOFT.NS","ARIHANTCAP.NS","GEOJITFSL.NS","GRAUWEIL.BO","MCLOUD.NS","LKPSEC.BO","TARACHAND.NS",
#   "CENTEXT.NS","IRISDOREME.NS","BLIL.BO","RNBDENIMS.BO","ONEPOINT.NS","SONAMLTD.NS","GATEWAY.NS","RSYSTEMS.BO","INDRAMEDCO.NS","JYOTHYLAB.NS",
#   "FCL.NS","MANINFRA.NS","GPIL.NS","JAGSNPHARM.NS","HSCL.NS","JWL.NS","BSOFT.NS","MARKSANS.NS","TALBROAUTO.NS","GALLANTT.NS",
#   "RESPONIND.NS","NAM-INDIA.NS","MONARCH.NS","ELECON.NS","SHANTIGEAR.NS","JASH.NS","GARFIBRES.NS","VISHNU.NS","GRSE.NS","RITES.NS",
#   "AEGISLOG.NS","ZENTEC.NS","DELHIVERY.NS","IFCI.NS","CDSL.NS","NUVAMA.NS","NEULANDLAB.NS","GODFRYPHLP.NS","BAJAJHFL.NS","HBLENGINE.NS",
#   "DLF.NS","RKFORGE.NS","EMIL.NS","JMFINANCIL.NS"
# ]
# [
#     "FILATFASH.NS", "SRESTHA.BO", "HARSHILAGR.BO", "GTLINFRA.NS", "ITC.NS", "OBEROIRLTY.NS",
#     "JAMNAAUTO.NS", "KSOLVES.NS", "ADANIGREEN.NS", "TATAMOTORS.NS", "OLECTRA.NS", "ARE&M.NS",
#     "AFFLE.NS", "BEL.NS", "SUNPHARMA.NS", "LAURUSLABS.NS", "RELIANCE.NS", "KRBL.NS", "ONGC.NS",
#     "IDFCFIRSTB.NS", "BANKBARODA.NS", "GSFC.NS", "TCS.NS", "INFY.NS", "SVARTCORP.BO", "SWASTIVI.BO",
#     "BTML.NS", "SULABEN.BO", "CRYSTAL.BO", "TILAK.BO", "COMFINTE.BO", "COCHINSHIP.NS", "RVNL.NS",
#     "SHAILY.NS", "BDL.NS", "JYOTICNC.NS", "NATIONALUM.NS", "KRONOX.NS", "SAKSOFT.NS", "ARIHANTCAP.NS",
#     "GEOJITFSL.NS", "GRAUWEIL.BO", "MCLOUD.NS", "LKPSEC.BO", "TARACHAND.NS", "CENTEXT.NS",
#     "IRISDOREME.NS", "BLIL.BO", "RNBDENIMS.BO", "ONEPOINT.NS", "SONAMLTD.NS", "GATEWAY.NS",
#     "RSYSTEMS.BO", "INDRAMEDCO.NS", "JYOTHYLAB.NS", "FCL.NS", "MANINFRA.NS", "GPIL.NS",
#     "JAGSNPHARM.NS", "HSCL.NS", "JWL.NS", "BSOFT.NS", "MARKSANS.NS", "TALBROAUTO.NS", "GALLANTT.NS",
#     "RESPONIND.NS", "IRCTC.NS", "NAM-INDIA.NS", "MONARCH.NS", "ELECON.NS", "SHANTIGEAR.NS",
#     "JASH.NS", "GARFIBRES.NS", "VISHNU.NS", "GRSE.NS", "RITES.NS", "AEGISLOG.NS", "ZENTEC.NS",
#     "DELHIVERY.NS", "IFCI.NS", "CDSL.NS", "NUVAMA.NS", "NEULANDLAB.NS", "GODFRYPHLP.NS",
#     "BAJAJHFL.NS", "PIDILITIND.NS", "HBLENGINE.NS", "DLF.NS", "RKFORGE.NS"
# ]

# [
#     "SRESTHA.BO",
#     "ITC.NS",
#     "JAMNAAUTO.NS",
#     "KSOLVES.NS",
#     "ADANIGREEN.NS",
#     "TATAMOTORS.NS",
#     "OLECTRA.NS",
#     "ARE&M.NS",
#     "AFFLE.NS",
#     "BEL.NS",
#     "LAURUSLABS.NS",
#     "IDFCFIRSTB.NS",
#     "GSFC.NS",
#     "INFY.NS",
#     "BTML.NS",
#     "CRYSTAL.BO",
#     "TILAK.BO",
#     "ARIHANTCAP.NS",
#     "LKPSEC.BO",
#     "CENTEXT.NS",
#     "RNBDENIMS.BO",
#     "RSYSTEMS.BO",
#     "INDRAMEDCO.NS",
#     "JAGSNPHARM.NS",
#     "HSCL.NS",
#     "TALBROAUTO.NS",
#     "GALLANTT.NS",
#     "ELECON.NS",
#     "ZENTEC.NS"
# ]


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

tickers_str = os.getenv("TICKERS")
TICKERS = tickers_str.split(",") if tickers_str else []


CHECK_INTERVAL = 60 * 5  # 5 minutes
SHARES_TO_BUY = 2
ATR_MULTIPLIER = 2.5 # 2.0  # Increased for better risk management
RSI_OVERSOLD = 20 # 25     # More strict oversold
RSI_OVERBOUGHT = 80 # 75   # More strict overbought

# NEW: Cooldown and risk management settings
TRADE_COOLDOWN_MINUTES = 30  # Minimum time between trades for same ticker
MAX_CONSECUTIVE_LOSSES = 3   # Stop trading ticker after consecutive losses
MIN_HOLDING_MINUTES = 15     # Minimum holding period
MAX_DAILY_TRADES = 40        # Maximum trades per day

# Advanced Configuration
MIN_VOLUME_SPIKE = 1.8  # Volume spike threshold
TREND_CONFIRMATION_PERIODS = 3  # Number of periods for trend confirmation
VOLATILITY_FILTER = 0.03  # Maximum daily volatility (3%)
CORRELATION_THRESHOLD = 0.7  # Market correlation threshold
STRENGTH_THRESHOLD = 65  # Relative strength threshold

# Market Hours (IST)
MARKET_START = "09:00"
MARKET_END = "16:00"
ALIVE_CHECK_MORNING = "09:15"
ALIVE_CHECK_EVENING = "15:00"

class ShutdownDiagnostics:
    def __init__(self, telegram_sender=None):
        self.start_time = datetime.now()
        self.send_telegram_message = telegram_sender or self.default_telegram
        self.setup_signal_handlers()
        self.log_startup_info()
    
    def default_telegram(self, message):
        """Default telegram sender (just print)"""
        print(f"[TELEGRAM] {message}")
    
    def setup_signal_handlers(self):
        """Enhanced signal handling with diagnostics"""
        def enhanced_signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.critical(f"SHUTDOWN INITIATED - Signal: {signal_name} ({signum})")
            
            # Log shutdown diagnostics
            self.log_shutdown_diagnostics(signum, frame)
            
            # Determine and log reason
            reason = self.determine_shutdown_reason(signum)
            logger.critical(f"LIKELY REASON: {reason}")
            
            # Send Telegram alert
            self.send_shutdown_alert(signal_name, reason)
            
            # Graceful cleanup
            self.cleanup_and_exit()
        
        try:
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, enhanced_signal_handler)
                logger.info("SIGTERM handler registered")
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, enhanced_signal_handler)
                logger.info("SIGINT handler registered")
        except ValueError as e:
            logger.warning(f"Signal handler setup failed: {e}")
    
    def log_startup_info(self):
        """Log startup information to console"""
        try:
            logger.info("=== STARTUP DIAGNOSTICS ===")
            logger.info(f"Process ID: {os.getpid()}")
            logger.info(f"Parent PID: {os.getppid()}")
            logger.info(f"User: {os.getenv('USER', 'unknown')}")
            logger.info(f"Working Directory: {os.getcwd()}")
            logger.info(f"In Container: {os.path.exists('/.dockerenv')}")
            logger.info(f"Start Time: {self.start_time}")
            
            # Log parent process info
            try:
                parent = psutil.Process(os.getppid())
                logger.info(f"Parent Process: {parent.name()} - {' '.join(parent.cmdline()[:3])}")
            except:
                logger.warning("Could not get parent process info")
            
            logger.info("=== END STARTUP DIAGNOSTICS ===")
            
        except Exception as e:
            logger.error(f"Error logging startup info: {e}")
    
    def log_shutdown_diagnostics(self, signum, frame):
        """Log shutdown diagnostics to console"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            signal_name = signal.Signals(signum).name if signum in signal.Signals._value2name_ else 'UNKNOWN'
            
            logger.info("=== SHUTDOWN DIAGNOSTICS ===")
            logger.info(f"Shutdown Time: {datetime.now()}")
            logger.info(f"Uptime: {uptime:.1f} seconds ({uptime/60:.1f} minutes)")
            logger.info(f"Signal: {signal_name} ({signum})")
            logger.info(f"Frame Location: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}()")
            
            # Log system resources at shutdown
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                disk = psutil.disk_usage('/')
                
                logger.info(f"Memory Usage: {memory.percent}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")
                logger.info(f"CPU Usage: {cpu}%")
                logger.info(f"Disk Usage: {disk.percent}%")
                
                if hasattr(os, 'getloadavg'):
                    load = os.getloadavg()
                    logger.info(f"Load Average: {load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}")
                
            except Exception as e:
                logger.warning(f"Could not get system resources: {e}")
            
            # Check for OOM killer traces (simple check)
            try:
                result = subprocess.run(['dmesg', '-T'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'killed process' in result.stdout.lower():
                    logger.warning("OOM Killer activity detected in system logs")
                else:
                    logger.info("No OOM Killer activity detected")
            except:
                logger.info("Could not check OOM killer status")
            
            logger.info("=== END SHUTDOWN DIAGNOSTICS ===")
            
        except Exception as e:
            logger.error(f"Error logging shutdown diagnostics: {e}")
    
    def determine_shutdown_reason(self, signum):
        """Analyze and determine likely shutdown reason"""
        reasons = []
        
        # Signal analysis
        if signum == signal.SIGTERM:
            reasons.append("SIGTERM received")
            
            # Check parent process
            try:
                parent = psutil.Process(os.getppid())
                parent_name = parent.name().lower()
                if 'gunicorn' in parent_name:
                    reasons.append("Gunicorn worker restart/reload")
                elif 'systemd' in parent_name:
                    reasons.append("Systemd service stop/restart")
                elif 'docker' in parent_name or 'containerd' in parent_name:
                    reasons.append("Container stop/restart")
                elif parent_name == 'init' or parent.pid == 1:
                    reasons.append("Process orphaned or system shutdown")
            except:
                reasons.append("Could not analyze parent process")
        
        elif signum == signal.SIGINT:
            reasons.append("SIGINT - Manual interruption (Ctrl+C)")
        
        # Time-based analysis
        uptime = (datetime.now() - self.start_time).total_seconds()
        if uptime < 60:
            reasons.append("Quick shutdown - startup failure possible")
        elif uptime > 28800:  # 8 hours
            reasons.append("Long-running - scheduled maintenance likely")
        
        # Container check
        if os.path.exists('/.dockerenv'):
            reasons.append("Container environment - check orchestration logs")
        
        return " | ".join(reasons) if reasons else "Unknown shutdown reason"
    
    def send_shutdown_alert(self, signal_name, reason):
        """Send shutdown alert via Telegram"""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        uptime_str = f"{int(uptime_seconds//3600)}h {int((uptime_seconds%3600)//60)}m {int(uptime_seconds%60)}s"
        
        message = f"🛑 TRADING BOT SHUTDOWN\n"
        message += f"📊 Signal: {signal_name}\n"
        message += f"⏱️ Uptime: {uptime_str}\n"
        message += f"🔍 Analysis: {reason}\n"
        message += f"📝 Check console logs for details"
        
        try:
            self.send_telegram_message(message)
            logger.info("Shutdown alert sent via Telegram")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def cleanup_and_exit(self):
        """Enhanced cleanup with logging"""
        logger.info("Starting graceful shutdown cleanup...")
        
        # Your existing cleanup code would go here
        # print_final_summary()  # Your existing function
        
        logger.info("Cleanup completed - goodbye!")
        sys.exit(0)
    
    def log_health_warning(self, metric, value):
        """Log health warnings and send alerts if needed"""
        message = f"⚠️ HIGH {metric.upper()}: {value}"
        logger.warning(message)
        
        # Send telegram alert for critical resource usage
        if (metric == "MEMORY" and value > 95) or (metric == "CPU" and value > 98):
            telegram_msg = f"🚨 CRITICAL RESOURCE ALERT\n{metric}: {value}%\nBot may shutdown soon!"
            try:
                self.send_telegram_message(telegram_msg)
            except:
                pass

# Initialize diagnostics (will be set up in your main code)
shutdown_diagnostics = None

def setup_enhanced_diagnostics(telegram_function=None):
    """Setup enhanced diagnostics - call this from your main code"""
    global shutdown_diagnostics
    shutdown_diagnostics = ShutdownDiagnostics(telegram_function)
    logger.info("Enhanced shutdown diagnostics initialized")

# Modified health monitoring function
# def monitor_system_health():
#     """Monitor system health and log warnings"""
#     try:
#         memory_percent = psutil.virtual_memory().percent
#         cpu_percent = psutil.cpu_percent(interval=1)
        
#         if memory_percent > 90:
#             shutdown_diagnostics.log_health_warning("MEMORY", f"{memory_percent:.1f}%")
#         if cpu_percent > 95:
#             shutdown_diagnostics.log_health_warning("CPU", f"{cpu_percent:.1f}%")
        
#         # Log uptime periodically
#         if shutdown_diagnostics:
#             uptime = datetime.now() - shutdown_diagnostics.start_time
#             if uptime.seconds % 3600 < 60:  # Every hour
#                 logger.info(f"Bot uptime: {int(uptime.total_seconds()/3600)}h {int((uptime.seconds%3600)/60)}m")
                
#     except Exception as e:
#         logger.error(f"Health monitoring error: {e}")

# def monitor_system_health():
#     """Monitor system health and log warnings with Telegram alerts"""
#     try:
#         memory_info = psutil.virtual_memory()
#         memory_percent = memory_info.percent
#         cpu_percent = psutil.cpu_percent(interval=1)
        
#         # Get current process memory usage in MB
#         process = psutil.Process()
#         process_memory_mb = process.memory_info().rss / 1024 / 1024
        
#         # Memory alerts
#         if memory_percent > 90:
#             shutdown_diagnostics.log_health_warning("SYSTEM_MEMORY", f"{memory_percent:.1f}%")
        
#         if cpu_percent > 95:
#             shutdown_diagnostics.log_health_warning("CPU", f"{cpu_percent:.1f}%")
        
#         # Process memory alert with Telegram notification
#         if process_memory_mb > 450:  # 450MB threshold
#             warning_msg = f"🚨 HIGH MEMORY USAGE ALERT 🚨\n"
#             warning_msg += f"💾 Bot Memory: {process_memory_mb:.1f}MB / 512MB\n"
#             warning_msg += f"📊 System Memory: {memory_percent:.1f}%\n"
#             warning_msg += f"⚡ CPU: {cpu_percent:.1f}%\n"
#             warning_msg += f"🧹 Triggering cleanup..."
            
#             send_telegram_message(warning_msg)
#             logger.warning(f"HIGH PROCESS MEMORY: {process_memory_mb:.1f}MB")
            
#             # Trigger immediate cleanup
#             cleanup_memory()
            
#             # Check memory again after cleanup
#             new_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
#             reduction = process_memory_mb - new_memory_mb
            
#             if reduction > 0:
#                 cleanup_msg = f"✅ Memory cleanup completed\n"
#                 cleanup_msg += f"📉 Freed: {reduction:.1f}MB\n"
#                 cleanup_msg += f"💾 Current: {new_memory_mb:.1f}MB"
#                 send_telegram_message(cleanup_msg)
        
#         # Log memory stats periodically
#         if shutdown_diagnostics:
#             uptime = datetime.now() - shutdown_diagnostics.start_time
#             if uptime.seconds % 3600 < 60:  # Every hour
#                 logger.info(f"Memory Status: Process={process_memory_mb:.1f}MB, System={memory_percent:.1f}%")
                
#     except Exception as e:
#         logger.error(f"Health monitoring error: {e}")

# def monitor_staged_system_health():
#     """Enhanced system health monitoring for staged loading"""
#     try:
#         memory_info = psutil.virtual_memory()
#         memory_percent = memory_info.percent
#         cpu_percent = psutil.cpu_percent(interval=1)
        
#         # Get current process memory usage
#         process = psutil.Process()
#         process_memory_mb = process.memory_info().rss / 1024 / 1024
        
#         # Get cache size
#         cache_size_mb = data_cache.get_cache_size_mb()
        
#         # System alerts
#         if memory_percent > 90:
#             shutdown_diagnostics.log_health_warning("SYSTEM_MEMORY", f"{memory_percent:.1f}%")
        
#         if cpu_percent > 95:
#             shutdown_diagnostics.log_health_warning("CPU", f"{cpu_percent:.1f}%")
        
#         # Process memory alert with staged loading context
#         if process_memory_mb > 400:  # Lower threshold for staged loading
#             warning_msg = f"🚨 STAGED LOADING MEMORY ALERT 🚨\n"
#             warning_msg += f"💾 Bot Memory: {process_memory_mb:.1f}MB / 512MB\n"
#             warning_msg += f"🗄️ Cache Size: {cache_size_mb:.1f}MB\n"
#             warning_msg += f"📊 System Memory: {memory_percent:.1f}%\n"
#             warning_msg += f"⚡ CPU: {cpu_percent:.1f}%\n"
#             warning_msg += f"🧹 Triggering enhanced cleanup..."
            
#             send_telegram_message(warning_msg)
#             logger.warning(f"HIGH STAGED PROCESS MEMORY: {process_memory_mb:.1f}MB")
            
#             # Trigger enhanced cleanup
#             enhanced_cleanup_memory()
            
#             # Check memory again after cleanup
#             new_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
#             new_cache_mb = data_cache.get_cache_size_mb()
#             reduction = process_memory_mb - new_memory_mb
            
#             if reduction > 0:
#                 cleanup_msg = f"✅ Staged cleanup completed\n"
#                 cleanup_msg += f"📉 Process freed: {reduction:.1f}MB\n"
#                 cleanup_msg += f"💾 Process current: {new_memory_mb:.1f}MB\n"
#                 cleanup_msg += f"🗄️ Cache current: {new_cache_mb:.1f}MB"
#                 send_telegram_message(cleanup_msg)
        
#         # Periodic memory logging
#         if shutdown_diagnostics:
#             uptime = datetime.now() - shutdown_diagnostics.start_time
#             if uptime.seconds % 3600 < 60:  # Every hour
#                 logger.info(f"Staged Memory Status: Process={process_memory_mb:.1f}MB, Cache={cache_size_mb:.1f}MB, System={memory_percent:.1f}%")
                
#     except Exception as e:
#         logger.error(f"Staged health monitoring error: {e}")

def monitor_staged_system_health():
    """Fixed system health monitoring"""
    try:
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        cpu_percent = psutil.cpu_percent(interval=1)
        
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # This line was causing crashes - add safety check:
        try:
            cache_size_mb = data_cache.get_cache_size_mb()
        except Exception as e:
            logger.warning(f"Could not get cache size: {e}")
            cache_size_mb = 0.0
        
        # Lower threshold for 512MB instance
        if process_memory_mb > 300:  # Much more conservative
            warning_msg = f"🚨 MEMORY ALERT 🚨\n"
            warning_msg += f"💾 Bot: {process_memory_mb:.1f}MB/512MB\n"
            warning_msg += f"🗄️ Cache: {cache_size_mb:.1f}MB\n"
            warning_msg += f"🧹 Cleanup triggered..."
            
            send_telegram_message(warning_msg)
            enhanced_cleanup_memory()
        
        # Log every hour
        if shutdown_diagnostics:
            uptime = datetime.now() - shutdown_diagnostics.start_time
            if uptime.seconds % 3600 < 60:
                logger.info(f"Memory: Process={process_memory_mb:.1f}MB, Cache={cache_size_mb:.1f}MB")
                
    except Exception as e:
        logger.error(f"Health monitoring error: {e}")

# Integration function for your existing code
def integrate_with_existing_bot():
    """Add this to your existing trading_loop.py"""
    print("""
    # Add these imports to your trading_loop.py:
    from enhanced_diagnostics import setup_enhanced_diagnostics, monitor_system_health
    
    # In your main() function, add:
    setup_enhanced_diagnostics(send_telegram_message)  # Pass your telegram function
    
    # In your main trading loop, add periodic health checks:
    if loop_count % 6 == 0:  # Every 30 minutes (assuming 5-min intervals)
        monitor_system_health()
    """)

class InMemoryStockMemory:
    def __init__(self):
        self.holdings = {}
        self.sell_thresholds = {}
        self.highest_prices = {}
        self.alerts_sent = {}
        self.last_action_status = {}
        self.price_history = {}
        self.volume_history = {}
        self.signal_strength = {}
        self.market_sentiment = 'NEUTRAL'
        self.correlation_matrix = {}
        self.last_alive_check = None
        self.session_start_time = datetime.now()
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        self.shutdown_flag = False
        
        # Enhanced tracking without persistence
        self.trade_cooldowns = {}  # {ticker: last_trade_time}
        self.consecutive_losses = {}  # {ticker: count}
        self.last_trade_time = {}  # {ticker: datetime}
        self.ticker_blacklist = set()  # Tickers to avoid
        
        # Initialize for all tickers
        for ticker in TICKERS:
            self.price_history[ticker] = deque(maxlen=20)
            self.volume_history[ticker] = deque(maxlen=20)
            self.alerts_sent[ticker] = {'52w_high': False, 'breakout': False, 'support': False}
            self.signal_strength[ticker] = 0.0
            self.trade_cooldowns[ticker] = None
            self.consecutive_losses[ticker] = 0
            self.last_trade_time[ticker] = None


    def is_in_cooldown(ticker: str) -> bool:
        """Check if ticker is in trading cooldown"""
        if ticker not in memory.trade_cooldowns or memory.trade_cooldowns[ticker] is None:
            return False
        
        cooldown_end = memory.trade_cooldowns[ticker] + timedelta(minutes=TRADE_COOLDOWN_MINUTES)
        return datetime.now() < cooldown_end

    def should_skip_ticker(ticker: str) -> tuple[bool, str]:
        """Enhanced ticker filtering"""
        
        # Check if ticker is in our approved list
        if ticker not in TICKERS:
            return True, "Not in approved ticker list"
        
        # Check blacklist
        if ticker in memory.ticker_blacklist:
            return True, "Blacklisted ticker"
        
        # Check cooldown
        if is_in_cooldown(ticker):
            remaining_minutes = int(((memory.trade_cooldowns[ticker] + 
                                    timedelta(minutes=TRADE_COOLDOWN_MINUTES)) - 
                                datetime.now()).total_seconds() / 60)
            return True, f"Cooldown ({remaining_minutes}min left)"
        
        # Check consecutive losses
        if memory.consecutive_losses.get(ticker, 0) >= MAX_CONSECUTIVE_LOSSES:
            return True, f"Too many losses ({memory.consecutive_losses[ticker]})"
        
        # Check daily trade limit
        if memory.total_trades >= MAX_DAILY_TRADES:
            return True, "Daily limit reached"
        
        return False, "OK"
# class OptimizedDataCache:
#     def __init__(self):
#         self.historical_cache = {}
#         self.cache_timestamps = {}
#         self.CACHE_EXPIRE_HOURS = 2  # 2 hour expiry
#         self.indicators_cache = {}
    
#     def get_cached_historical(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
#         """Get cached historical data or fetch if expired"""
#         now = datetime.now()
#         cache_key = f"{ticker}_{period}"
        
#         # Check if cache exists and is fresh
#         if (cache_key in self.historical_cache and 
#             cache_key in self.cache_timestamps and
#             (now - self.cache_timestamps[cache_key]).total_seconds() < self.CACHE_EXPIRE_HOURS * 3600):
#             logger.info(f"Using cached data for {ticker}")
#             return self.historical_cache[cache_key]
        
#         # Fetch fresh data
#         logger.info(f"Fetching fresh data for {ticker}")
#         df = get_stock_data(ticker, period)
#         if df is not None:
#             self.historical_cache[cache_key] = df
#             self.cache_timestamps[cache_key] = now
        
#         return df
    
#     def get_cached_indicators(self, ticker: str, df: pd.DataFrame) -> Dict:
#         """Get cached indicators or calculate if expired"""
#         now = datetime.now()
#         cache_key = f"{ticker}_indicators"
        
#         if (cache_key in self.indicators_cache and 
#             cache_key in self.cache_timestamps and
#             (now - self.cache_timestamps[cache_key]).total_seconds() < self.CACHE_EXPIRE_HOURS * 3600):
#             return self.indicators_cache[cache_key]
        
#         # Calculate fresh indicators
#         indicators = calculate_advanced_indicators(df)
#         if indicators:
#             self.indicators_cache[cache_key] = indicators
#             self.cache_timestamps[cache_key] = now
        
#         return indicators

# class OptimizedDataCache:
#     def __init__(self):
#         self.historical_cache = {}
#         self.cache_timestamps = {}
#         self.indicators_cache = {}
#         self.CACHE_EXPIRE_HOURS = 2
#         self.MAX_CACHE_SIZE_MB = 80  # Conservative limit for 512MB instance
        
#     def get_cache_size_mb(self) -> float:
#         """Calculate current cache size in MB - CRITICAL METHOD"""
#         try:
#             import sys
#             total_size = 0
            
#             # Calculate historical cache size
#             for key, df in self.historical_cache.items():
#                 if df is not None:
#                     total_size += df.memory_usage(deep=True).sum()
            
#             # Calculate indicators cache size  
#             for key, indicators in self.indicators_cache.items():
#                 if indicators:
#                     total_size += sys.getsizeof(indicators)
#                     for k, v in indicators.items():
#                         if hasattr(v, 'nbytes'):  # numpy array
#                             total_size += v.nbytes
#                         else:
#                             total_size += sys.getsizeof(v)
            
#             return total_size / (1024 * 1024)
#         except Exception as e:
#             logger.warning(f"Error calculating cache size: {e}")
#             return 0.0

#     def cleanup_old_cache_entries(self):
#         """Remove expired cache entries - CRITICAL METHOD"""
#         try:
#             now = datetime.now()
#             cutoff = now - timedelta(hours=self.CACHE_EXPIRE_HOURS)
            
#             expired_keys = [key for key, timestamp in self.cache_timestamps.items() 
#                           if timestamp < cutoff]
            
#             cleaned_count = 0
#             for key in expired_keys:
#                 if key in self.historical_cache:
#                     del self.historical_cache[key]
#                     cleaned_count += 1
#                 if key in self.indicators_cache:
#                     del self.indicators_cache[key]
#                     cleaned_count += 1
#                 if key in self.cache_timestamps:
#                     del self.cache_timestamps[key]
            
#             if cleaned_count > 0:
#                 logger.info(f"Cache cleanup: Removed {cleaned_count} expired entries")
                
#         except Exception as e:
#             logger.error(f"Error in cache cleanup: {e}")
    
#     def force_cleanup_if_needed(self):
#         """Force cleanup if cache exceeds limit - CRITICAL METHOD"""
#         current_size = self.get_cache_size_mb()
#         if current_size > self.MAX_CACHE_SIZE_MB:
#             logger.warning(f"Cache size {current_size:.1f}MB exceeds limit - forcing cleanup")
            
#             if self.cache_timestamps:
#                 sorted_entries = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
#                 entries_to_remove = len(sorted_entries) // 2  # Remove oldest 50%
                
#                 for key, _ in sorted_entries[:entries_to_remove]:
#                     if key in self.historical_cache:
#                         del self.historical_cache[key]
#                     if key in self.indicators_cache:
#                         del self.indicators_cache[key]
#                     if key in self.cache_timestamps:
#                         del self.cache_timestamps[key]
                
#                 new_size = self.get_cache_size_mb()
#                 logger.info(f"Forced cleanup: Size reduced from {current_size:.1f}MB to {new_size:.1f}MB")

#     def get_cached_historical(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
#         """Get cached historical data with memory management"""
#         now = datetime.now()
#         cache_key = f"{ticker}_{period}"
        
#         # Check cache first
#         if (cache_key in self.historical_cache and 
#             cache_key in self.cache_timestamps and
#             (now - self.cache_timestamps[cache_key]).total_seconds() < self.CACHE_EXPIRE_HOURS * 3600):
#             logger.debug(f"Using cached data for {ticker}")
#             return self.historical_cache[cache_key]
        
#         # Force cleanup before fetching new data
#         self.force_cleanup_if_needed()
        
#         # Fetch fresh data
#         logger.info(f"Fetching fresh data for {ticker}")
#         df = get_stock_data(ticker, period)
#         if df is not None:
#             self.historical_cache[cache_key] = df
#             self.cache_timestamps[cache_key] = now
        
#         return df
    
#     def get_cached_indicators(self, ticker: str, df: pd.DataFrame) -> Dict:
#         """Get cached indicators with compression - CRITICAL FIX"""
#         now = datetime.now()
#         cache_key = f"{ticker}_indicators"
        
#         # Check cache
#         if (cache_key in self.indicators_cache and 
#             cache_key in self.cache_timestamps and
#             (now - self.cache_timestamps[cache_key]).total_seconds() < self.CACHE_EXPIRE_HOURS * 3600):
#             logger.debug(f"Using cached indicators for {ticker}")
#             return self.indicators_cache[cache_key]
        
#         # Calculate fresh indicators
#         logger.debug(f"Calculating indicators for {ticker}")
#         indicators = calculate_advanced_indicators(df)
#         if indicators:
#             # CRITICAL: Compress indicators to single values
#             compressed_indicators = {}
#             for key, value in indicators.items():
#                 if value is not None:
#                     latest_value = safe_extract(value)
#                     if latest_value is not None:
#                         compressed_indicators[key] = latest_value
            
#             self.indicators_cache[cache_key] = compressed_indicators
#             self.cache_timestamps[cache_key] = now
            
#             logger.debug(f"Cached compressed indicators for {ticker}: {len(compressed_indicators)} values")
        
#         return indicators  # Return full for immediate use, but cache compressed


class OptimizedDataCache:
    def __init__(self):
        self.historical_cache = {}
        self.cache_timestamps = {}
        self.indicators_cache = {}
        self.CACHE_EXPIRE_HOURS = 2
        self.MAX_CACHE_SIZE_MB = 80

    def get_cache_size_mb(self) -> float:
        """Calculate current cache size in MB"""
        try:
            import sys
            total_size = 0
            
            for key, df in self.historical_cache.items():
                if df is not None:
                    total_size += df.memory_usage(deep=True).sum()
            
            for key, indicators in self.indicators_cache.items():
                if indicators:
                    total_size += sys.getsizeof(indicators)
                    for k, v in indicators.items():
                        if hasattr(v, 'nbytes'):
                            total_size += v.nbytes
                        else:
                            total_size += sys.getsizeof(v)
            
            return total_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Error calculating cache size: {e}")
            return 0.0

    def cleanup_old_cache_entries(self):
        """Remove expired cache entries"""
        try:
            now = datetime.now()
            cutoff = now - timedelta(hours=self.CACHE_EXPIRE_HOURS)
            
            expired_keys = [key for key, timestamp in self.cache_timestamps.items() 
                          if timestamp < cutoff]
            
            cleaned_count = 0
            for key in expired_keys:
                if key in self.historical_cache:
                    del self.historical_cache[key]
                    cleaned_count += 1
                if key in self.indicators_cache:
                    del self.indicators_cache[key]
                    cleaned_count += 1
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
            
            if cleaned_count > 0:
                logger.info(f"Cache cleanup: Removed {cleaned_count} expired entries")
                
        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}")
    
    def force_cleanup_if_needed(self):
        """Force cleanup if cache exceeds limit"""
        current_size = self.get_cache_size_mb()
        if current_size > self.MAX_CACHE_SIZE_MB:
            logger.warning(f"Cache size {current_size:.1f}MB exceeds limit - forcing cleanup")
            
            if self.cache_timestamps:
                sorted_entries = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
                entries_to_remove = len(sorted_entries) // 2
                
                for key, _ in sorted_entries[:entries_to_remove]:
                    if key in self.historical_cache:
                        del self.historical_cache[key]
                    if key in self.indicators_cache:
                        del self.indicators_cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]
                
                new_size = self.get_cache_size_mb()
                logger.info(f"Forced cleanup: Size reduced from {current_size:.1f}MB to {new_size:.1f}MB")

    def get_cached_historical(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Get cached historical data with memory management"""
        now = datetime.now()
        cache_key = f"{ticker}_{period}"
        
        if (cache_key in self.historical_cache and 
            cache_key in self.cache_timestamps and
            (now - self.cache_timestamps[cache_key]).total_seconds() < self.CACHE_EXPIRE_HOURS * 3600):
            logger.debug(f"Using cached data for {ticker}")
            return self.historical_cache[cache_key]
        
        self.force_cleanup_if_needed()
        
        logger.info(f"Fetching fresh data for {ticker}")
        df = get_stock_data(ticker, period)
        if df is not None:
            self.historical_cache[cache_key] = df
            self.cache_timestamps[cache_key] = now
        
        return df
    
    def get_cached_indicators(self, ticker: str, df: pd.DataFrame) -> Dict:
        """Get cached indicators - FIXED VERSION"""
        now = datetime.now()
        cache_key = f"{ticker}_indicators"
        
        # Check cache first
        if (cache_key in self.indicators_cache and 
            cache_key in self.cache_timestamps and
            (now - self.cache_timestamps[cache_key]).total_seconds() < self.CACHE_EXPIRE_HOURS * 3600):
            logger.debug(f"Using cached indicators for {ticker}")
            return self.indicators_cache[cache_key]  # Return cached compressed indicators
        
        # Calculate fresh indicators
        logger.debug(f"Calculating indicators for {ticker}")
        raw_indicators = calculate_advanced_indicators(df)
        
        if raw_indicators:
            # Compress indicators to single values
            compressed_indicators = {}
            for key, value in raw_indicators.items():
                if value is not None:
                    try:
                        latest_value = safe_extract(value)
                        if latest_value is not None:
                            compressed_indicators[key] = latest_value
                    except Exception as e:
                        logger.debug(f"Error extracting {key} for {ticker}: {e}")
                        continue
            
            # Cache compressed version
            self.indicators_cache[cache_key] = compressed_indicators
            self.cache_timestamps[cache_key] = now
            
            logger.debug(f"Cached {len(compressed_indicators)} compressed indicators for {ticker}")
            
            # ✅ CRITICAL FIX: Return compressed indicators, not raw arrays
            return compressed_indicators
        
        return {}

memory = InMemoryStockMemory()
data_cache = OptimizedDataCache()  # Add this line

def cleanup_memory():
    """Periodic memory cleanup every 2 hours"""
    try:
        cutoff = datetime.now() - timedelta(hours=2)
        
        # Clear expired historical data cache
        expired_historical = [key for key, timestamp in data_cache.cache_timestamps.items() 
                            if timestamp < cutoff and '_indicators' not in key]
        
        # Clear expired indicators cache
        expired_indicators = [key for key, timestamp in data_cache.cache_timestamps.items() 
                            if timestamp < cutoff and '_indicators' in key]
        
        # Remove expired entries
        for key in expired_historical:
            if key in data_cache.historical_cache:
                del data_cache.historical_cache[key]
            if key in data_cache.cache_timestamps:
                del data_cache.cache_timestamps[key]
        
        for key in expired_indicators:
            if key in data_cache.indicators_cache:
                del data_cache.indicators_cache[key]
            if key in data_cache.cache_timestamps:
                del data_cache.cache_timestamps[key]
        
        # Clear old price/volume history beyond 10 items
        for ticker in TICKERS:
            if ticker in memory.price_history and len(memory.price_history[ticker]) > 10:
                # Keep only last 10 items
                while len(memory.price_history[ticker]) > 10:
                    memory.price_history[ticker].popleft()
            
            if ticker in memory.volume_history and len(memory.volume_history[ticker]) > 10:
                while len(memory.volume_history[ticker]) > 10:
                    memory.volume_history[ticker].popleft()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        total_cleaned = len(expired_historical) + len(expired_indicators)
        if total_cleaned > 0:
            logger.info(f"Memory cleanup: Removed {total_cleaned} expired cache entries")
            
    except Exception as e:
        logger.error(f"Error in memory cleanup: {e}")   

def enhanced_cleanup_memory():
    """Enhanced memory cleanup for staged loading"""
    try:
        # Original cleanup
        cleanup_memory()
        
        # Enhanced cache cleanup
        data_cache.cleanup_old_cache_entries()
        data_cache.force_cleanup_if_needed()
        
        # Clear old deque data
        cutoff_time = datetime.now() - timedelta(hours=1)
        for ticker in TICKERS:
            # Keep only recent price/volume history
            if ticker in memory.price_history:
                while (len(memory.price_history[ticker]) > 5):
                    memory.price_history[ticker].popleft()
            
            if ticker in memory.volume_history:
                while (len(memory.volume_history[ticker]) > 5):
                    memory.volume_history[ticker].popleft()
        
        # Reset alerts for better memory usage
        for ticker in TICKERS:
            if ticker in memory.alerts_sent:
                # Reset breakout alerts after 2 hours
                if all(memory.alerts_sent[ticker].values()):
                    memory.alerts_sent[ticker] = {'52w_high': False, 'breakout': False, 'support': False}
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Log memory status after cleanup
        cache_size = data_cache.get_cache_size_mb()
        process_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Enhanced cleanup completed: Process={process_memory:.1f}MB, Cache={cache_size:.1f}MB")
        
    except Exception as e:
        logger.error(f"Error in enhanced memory cleanup: {e}")

def is_in_cooldown(ticker: str) -> bool:
    """Check if ticker is in trading cooldown"""
    if ticker not in memory.trade_cooldowns or memory.trade_cooldowns[ticker] is None:
        return False
    
    cooldown_end = memory.trade_cooldowns[ticker] + timedelta(minutes=TRADE_COOLDOWN_MINUTES)
    return datetime.now() < cooldown_end

def should_skip_ticker(ticker: str) -> Tuple[bool, str]:
    """Check if ticker should be skipped due to risk management rules"""
    
    # Check cooldown
    if is_in_cooldown(ticker):
        remaining_minutes = ((memory.trade_cooldowns[ticker] + timedelta(minutes=TRADE_COOLDOWN_MINUTES)) - datetime.now()).seconds // 60
        return True, f"Cooldown ({remaining_minutes}min left)"
    
    # Check consecutive losses
    if memory.consecutive_losses.get(ticker, 0) >= MAX_CONSECUTIVE_LOSSES:
        return True, f"Too many losses ({memory.consecutive_losses[ticker]})"
    
    # Check daily trade limit
    if memory.total_trades >= MAX_DAILY_TRADES:
        return True, "Daily limit reached"
    
    # Check if ticker is even in our list
    if ticker not in TICKERS:
        return True, "Not in ticker list"
    
    return False, "OK"

# ============================
# EXIT HANDLERS
# ============================

def cleanup_and_exit():
    """Clean exit with summary"""
    print("\nBot shutting down...")
    print_final_summary()
    send_telegram_message("🛑 *Bot Stopped*\nTrading session ended")
    memory.shutdown_flag = True
    sys.exit(0)

def setup_exit_handlers():
    """Setup graceful exit handlers"""
    def signal_handler(sig, frame):
        cleanup_and_exit()
    
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    # atexit.register(print_final_summary)

    try:
        # Only set up signal handlers if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("Signal handlers set up successfully")
        else:
            logger.warning("Not in main thread, skipping signal handler setup")
    except ValueError as e:
        logger.warning(f"Could not set up signal handlers: {e}")
    
    # Always set up atexit handler
    atexit.register(print_final_summary)

def print_final_summary():
    """Print final session summary"""
    try:
        session_duration = datetime.now() - memory.session_start_time
        active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
        
        print("\n" + "="*80)
        print("ADVANCED TRADING SESSION SUMMARY")
        print("="*80)
        print(f"Session Duration: {session_duration}")
        print(f"Total Trades: {memory.total_trades}")
        print(f"Profitable Trades: {memory.profitable_trades}")
        print(f"Win Rate: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "Win Rate: 0%")
        print(f"Total P&L: Rs.{memory.total_pnl:.2f}")
        print(f"Max Drawdown: {memory.max_drawdown:.2f}%")
        print(f"Active Positions: {active_positions}")
        print(f"Market Sentiment: {memory.market_sentiment}")
        print("="*80)
    except Exception as e:
        print(f"Error in final summary: {e}")

# ============================
# TELEGRAM FUNCTIONS
# ============================

def send_telegram_message(message: str):
    """Send message to all configured Telegram chats"""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print(f"[TELEGRAM] {message}")
        return

    # Parse TELEGRAM_CHAT_ID if it's a string
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

def send_alive_notification():
    """Send enhanced bot alive notification"""
    current_time = datetime.now().strftime("%H:%M")
    active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
    strong_signals = sum(1 for strength in memory.signal_strength.values() if strength > 70)
    
    message = f" *Advanced Stock Bot ALIVE* - {current_time}\n"
    message += f" Monitoring {len(TICKERS)} stocks\n"
    message += f" Active positions: {active_positions}\n"
    message += f" Strong signals: {strong_signals}\n"
    message += f" Market sentiment: {memory.market_sentiment}\n"
    message += f" Session P&L: Rs.{memory.total_pnl:.2f}"
    
    send_telegram_message(message)
    memory.last_alive_check = datetime.now()

# ============================
# ADVANCED TECHNICAL INDICATORS
# ============================

def calculate_advanced_indicators(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive technical indicators"""
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
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
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

# def calculate_signal_strength(ticker: str, indicators: Dict, current_price: float) -> float:
#     """Calculate overall signal strength (0-100)"""
#     try:
#         if not indicators:
#             return 0.0
        
#         strength_components = []
        
#         # Trend strength (30% weight)
#         sma_20 = safe_extract(indicators.get('sma_20'))
#         sma_50 = safe_extract(indicators.get('sma_50'))
#         ema_12 = safe_extract(indicators.get('ema_12'))
#         ema_26 = safe_extract(indicators.get('ema_26'))
        
#         if all([sma_20, sma_50, ema_12, ema_26]):
#             trend_score = 0
#             if current_price > sma_20 > sma_50:  # Strong uptrend
#                 trend_score += 25
#             if ema_12 > ema_26:  # EMA bullish crossover
#                 trend_score += 15
#             if current_price > sma_20:  # Above short-term MA
#                 trend_score += 10
#             strength_components.append(min(trend_score, 30))
        
#         # Momentum strength (25% weight)
#         rsi = safe_extract(indicators.get('rsi_14'))
#         macd = safe_extract(indicators.get('macd'))
#         macd_signal = safe_extract(indicators.get('macd_signal'))
        
#         if rsi and macd and macd_signal:
#             momentum_score = 0
#             if 30 < rsi < 70:  # RSI in good range
#                 momentum_score += 15
#             if macd > macd_signal and macd > 0:  # MACD bullish
#                 momentum_score += 10
#             strength_components.append(min(momentum_score, 25))
        
#         # Volume strength (20% weight) - FIXED SECTION
#         volume_sma = safe_extract(indicators.get('volume_sma_10'))
#         if volume_sma and ticker in memory.volume_history:
#             volume_history = memory.volume_history[ticker]
            
#             # Handle both scalar and array cases
#             if hasattr(volume_history, '__len__') and not isinstance(volume_history, str):
#                 # It's an array/list
#                 if len(volume_history) > 0:
#                     current_volume = volume_history[-1]
#                 else:
#                     current_volume = None
#             else:
#                 # It's a scalar value
#                 current_volume = volume_history
            
#             if current_volume is not None:
#                 volume_score = 0
#                 if current_volume > volume_sma * MIN_VOLUME_SPIKE:  # Volume spike
#                     volume_score += 20
#                 elif current_volume > volume_sma * 1.2:  # Good volume
#                     volume_score += 10
#                 strength_components.append(min(volume_score, 20))
        
#         # Volatility check (15% weight)
#         atr = safe_extract(indicators.get('atr_14'))
#         volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
#         if atr and volatility_ratio:
#             volatility_score = 0
#             if volatility_ratio < 1.5:  # Low volatility is good
#                 volatility_score += 15
#             elif volatility_ratio < 2.0:
#                 volatility_score += 10
#             strength_components.append(min(volatility_score, 15))
        
#         # Support/Resistance (10% weight)
#         bb_lower = safe_extract(indicators.get('bb_lower'))
#         bb_upper = safe_extract(indicators.get('bb_upper'))
#         if bb_lower and bb_upper:
#             sr_score = 0
#             if current_price > bb_lower and current_price < bb_upper:  # Within bands
#                 sr_score += 5
#             if current_price > bb_lower * 1.02:  # Above support with buffer
#                 sr_score += 5
#             strength_components.append(min(sr_score, 10))
        
#         # Calculate weighted average
#         total_strength = sum(strength_components)
#         return min(total_strength, 100.0)
        
#     except Exception as e:
#         print(f"Error calculating signal strength for {ticker}: {e}")
#         import traceback
#         print(f"Traceback: {traceback.format_exc()}")
#         return 0.0

# def calculate_signal_strength(ticker: str, indicators: Dict, current_price: float) -> float:
#     """Calculate signal strength using cached single values"""
#     try:
#         if not indicators:
#             return 0.0
        
#         strength_components = []
        
#         # All indicators are now single values from cache
#         sma_20 = indicators.get('sma_20')  # No need for safe_extract
#         sma_50 = indicators.get('sma_50') 
#         ema_12 = indicators.get('ema_12')
#         ema_26 = indicators.get('ema_26')
#         rsi = indicators.get('rsi_14')
#         macd = indicators.get('macd')
#         macd_signal = indicators.get('macd_signal')
        
#         # Trend strength (30% weight)
#         if all([sma_20, sma_50, ema_12, ema_26]):
#             trend_score = 0
#             if current_price > sma_20 > sma_50:
#                 trend_score += 25
#             if ema_12 > ema_26:
#                 trend_score += 15
#             if current_price > sma_20:
#                 trend_score += 10
#             strength_components.append(min(trend_score, 30))
        
#         # Momentum strength (25% weight)
#         if rsi and macd and macd_signal:
#             momentum_score = 0
#             if 30 < rsi < 70:
#                 momentum_score += 15
#             if macd > macd_signal and macd > 0:
#                 momentum_score += 10
#             strength_components.append(min(momentum_score, 25))
        
#         # # Volume strength (20% weight)
#         # volume_sma = indicators.get('volume_sma_10')
#         # if volume_sma and ticker in memory.volume_history and len(memory.volume_history[ticker]) > 0:
#         #     current_volume = memory.volume_history[ticker][-1]
#         #     volume_score = 0
#         #     if current_volume > volume_sma * MIN_VOLUME_SPIKE:
#         #         volume_score += 20
#         #     elif current_volume > volume_sma * 1.2:
#         #         volume_score += 10
#         #     strength_components.append(min(volume_score, 20))
        
#                 # Volume strength (20% weight) - FIXED SECTION
#         volume_sma = indicators.get('volume_sma_10')
#         if volume_sma is not None and ticker in memory.volume_history:
#             try:
#                 volume_history = memory.volume_history[ticker]
#                 current_volume = None
                
#                 # Safe extraction of current volume
#                 if hasattr(volume_history, '__len__') and len(volume_history) > 0:
#                     # It's a collection (list, deque, array)
#                     try:
#                         current_volume = float(volume_history[-1])
#                     except (IndexError, TypeError, ValueError):
#                         current_volume = None
#                 elif volume_history is not None:
#                     # It's a scalar
#                     try:
#                         current_volume = float(volume_history)
#                     except (TypeError, ValueError):
#                         current_volume = None
                
#                 if current_volume is not None and current_volume > 0 and volume_sma > 0:
#                     volume_score = 0
#                     volume_ratio = current_volume / volume_sma
                    
#                     if volume_ratio > MIN_VOLUME_SPIKE:
#                         volume_score += 20
#                     elif volume_ratio > 1.2:
#                         volume_score += 10
                    
#                     strength_components.append(min(volume_score, 20))
                    
#             except Exception as e:
#                 logger.debug(f"Volume analysis error for {ticker}: {e}")
#                 # Skip volume component if error occurs
#                 pass

#         # Volatility check (15% weight)
#         atr = indicators.get('atr_14')
#         volatility_ratio = indicators.get('volatility_ratio', 1.0)
#         if atr and volatility_ratio:
#             volatility_score = 0
#             if volatility_ratio < 1.5:
#                 volatility_score += 15
#             elif volatility_ratio < 2.0:
#                 volatility_score += 10
#             strength_components.append(min(volatility_score, 15))
        
#         # Support/Resistance (10% weight)
#         bb_lower = indicators.get('bb_lower')
#         bb_upper = indicators.get('bb_upper')
#         if bb_lower and bb_upper:
#             sr_score = 0
#             if current_price > bb_lower and current_price < bb_upper:
#                 sr_score += 5
#             if current_price > bb_lower * 1.02:
#                 sr_score += 5
#             strength_components.append(min(sr_score, 10))
        
#         return min(sum(strength_components), 100.0)
        
#     except Exception as e:
#         logger.error(f"Error calculating cached signal strength for {ticker}: {e}")
#         return 0.0

def calculate_signal_strength(ticker: str, indicators: Dict, current_price: float) -> float:
    """Calculate signal strength using compressed single values - FINAL VERSION"""
    try:
        if not indicators:
            return 0.0
        
        strength_components = []
        
        # All indicators are now guaranteed to be single values from cache
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50') 
        ema_12 = indicators.get('ema_12')
        ema_26 = indicators.get('ema_26')
        rsi = indicators.get('rsi_14')
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        
        # Trend strength (30% weight)
        if all(x is not None for x in [sma_20, sma_50, ema_12, ema_26]):
            trend_score = 0
            if current_price > sma_20 > sma_50:
                trend_score += 25
            if ema_12 > ema_26:
                trend_score += 15
            if current_price > sma_20:
                trend_score += 10
            strength_components.append(min(trend_score, 30))
        
        # Momentum strength (25% weight)
        if all(x is not None for x in [rsi, macd, macd_signal]):
            momentum_score = 0
            if 30 < rsi < 70:
                momentum_score += 15
            if macd > macd_signal and macd > 0:
                momentum_score += 10
            strength_components.append(min(momentum_score, 25))
        
        # Volume strength (20% weight) - SAFE VERSION
        volume_sma = indicators.get('volume_sma_10')
        if volume_sma is not None and ticker in memory.volume_history:
            try:
                volume_history = memory.volume_history[ticker]
                current_volume = None
                
                if hasattr(volume_history, '__len__') and len(volume_history) > 0:
                    try:
                        current_volume = float(volume_history[-1])
                    except (IndexError, TypeError, ValueError):
                        current_volume = None
                elif volume_history is not None:
                    try:
                        current_volume = float(volume_history)
                    except (TypeError, ValueError):
                        current_volume = None
                
                if current_volume is not None and current_volume > 0 and volume_sma > 0:
                    volume_score = 0
                    volume_ratio = current_volume / volume_sma
                    
                    if volume_ratio > MIN_VOLUME_SPIKE:
                        volume_score += 20
                    elif volume_ratio > 1.2:
                        volume_score += 10
                    
                    strength_components.append(min(volume_score, 20))
                    
            except Exception as e:
                logger.debug(f"Volume analysis error for {ticker}: {e}")
                pass
        
        # Volatility check (15% weight)
        atr = indicators.get('atr_14')
        volatility_ratio = indicators.get('volatility_ratio', 1.0)
        if atr is not None and volatility_ratio is not None:
            volatility_score = 0
            if volatility_ratio < 1.5:
                volatility_score += 15
            elif volatility_ratio < 2.0:
                volatility_score += 10
            strength_components.append(min(volatility_score, 15))
        
        # Support/Resistance (10% weight)
        bb_lower = indicators.get('bb_lower')
        bb_upper = indicators.get('bb_upper')
        if bb_lower is not None and bb_upper is not None:
            try:
                sr_score = 0
                if bb_lower < current_price < bb_upper:
                    sr_score += 5
                if current_price > bb_lower * 1.02:
                    sr_score += 5
                strength_components.append(min(sr_score, 10))
            except Exception as e:
                logger.debug(f"Support/Resistance analysis error for {ticker}: {e}")
                pass
        
        return min(sum(strength_components), 100.0)
        
    except Exception as e:
        logger.error(f"Error calculating signal strength for {ticker}: {e}")
        return 0.0

def calculate_market_sentiment() -> str:
    """Calculate overall market sentiment based on all stocks"""
    try:
        bullish_count = 0
        bearish_count = 0
        total_strength = 0
        
        for ticker in TICKERS:
            strength = memory.signal_strength.get(ticker, 0)
            total_strength += strength
            
            if strength > 60:
                bullish_count += 1
            elif strength < 40:
                bearish_count += 1
        
        avg_strength = total_strength / len(TICKERS) if TICKERS else 50
        
        if bullish_count >= len(TICKERS) * 0.6 or avg_strength > 65:
            return 'BULLISH'
        elif bearish_count >= len(TICKERS) * 0.6 or avg_strength < 35:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
            
    except Exception as e:
        print(f"Error calculating market sentiment: {e}")
        return 'NEUTRAL'

# def safe_extract(value, fallback=None):
#     """
#     Safely extract a numeric value from various data types
    
#     Args:
#         value: The value to extract from (Series, array, list, scalar)
#         fallback: Optional fallback value if extraction fails
    
#     Returns:
#         Extracted float value or fallback/None
#     """
#     if value is None:
#         return fallback
    
#     try:
#         # Handle pandas Series
#         if hasattr(value, 'iloc') :
#             if len(value) > 0:
#                 result = float(value.iloc[-1])
#                 return result if not (hasattr(result, '__iter__') or 
#                                     (hasattr(result, 'shape') and result.shape)) else fallback
#             else:
#                 return fallback
        
#         # Handle numpy arrays
#         elif hasattr(value, 'shape'):
#             if value.size > 0:
#                 result = float(value.flat[-1])  # Use .flat to handle any shape
#                 return result if not (hasattr(result, '__iter__') or 
#                                     (hasattr(result, 'shape') and result.shape)) else fallback
#             else:
#                 return fallback
        
#         # Handle lists and tuples
#         elif isinstance(value, (list, tuple)):
#             if len(value) > 0:
#                 return float(value[-1])
#             else:
#                 return fallback
        
#         # Handle scalar values (including numpy scalars)
#         else:
#             try:
#                 result = float(value)
#                 # Check if it's a valid number
#                 if hasattr(result, '__iter__'):  # Still an array somehow
#                     return fallback
#                 return result
#             except (ValueError, TypeError):
#                 return fallback
            
#     except (ValueError, TypeError, IndexError, AttributeError) as e:
#         print(f"Warning: safe_extract failed for value {type(value)}: {e}")
#         return fallback

def safe_extract(value, fallback=None):
    """
    Enhanced safe_extract that handles all edge cases properly
    """
    if value is None:
        return fallback
    
    try:
        # Handle pandas Series
        if hasattr(value, 'iloc') and hasattr(value, '__len__'):
            if len(value) > 0:
                extracted = value.iloc[-1]
                # Check if extracted value is still an array/series
                if hasattr(extracted, '__iter__') and not isinstance(extracted, (str, bytes)):
                    return fallback
                return float(extracted)
            return fallback
        
        # Handle numpy arrays
        elif hasattr(value, 'shape') and hasattr(value, 'size'):
            if value.size > 0:
                try:
                    extracted = value.flat[-1]
                    if hasattr(extracted, '__iter__') and not isinstance(extracted, (str, bytes)):
                        return fallback
                    return float(extracted)
                except (IndexError, ValueError):
                    return fallback
            return fallback
        
        # Handle lists and tuples
        elif isinstance(value, (list, tuple)):
            if len(value) > 0:
                try:
                    return float(value[-1])
                except (ValueError, TypeError):
                    return fallback
            return fallback
        
        # Handle scalar values
        else:
            try:
                result = float(value)
                # Additional check for numpy scalars that might still be arrays
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    return fallback
                return result
            except (ValueError, TypeError):
                return fallback
            
    except Exception as e:
        logger.debug(f"safe_extract_fixed error: {e}")
        return fallback

# ============================
# ENHANCED DATA FETCHING
# ============================

def get_stock_data(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """Fetch enhanced stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            print(f"No data for {ticker}")
            return None
        
        # Add additional calculated columns
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_realtime_data(ticker: str) -> Optional[Dict]:
    """Get enhanced real-time stock data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2d", interval="1m")
        if df.empty:
            return None
        
        current_price = df['Close'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        
        # Update price and volume history
        memory.price_history[ticker].append(current_price)
        memory.volume_history[ticker].append(current_volume)
        
        # Calculate intraday metrics
        day_high = df['High'].iloc[-390:].max() if len(df) >= 390 else df['High'].max()
        day_low = df['Low'].iloc[-390:].min() if len(df) >= 390 else df['Low'].min()
        day_open = df['Open'].iloc[-390] if len(df) >= 390 else df['Open'].iloc[0]
        
        return {
            'price': current_price,
            'volume': current_volume,
            'high': df['High'].iloc[-1],
            'low': df['Low'].iloc[-1],
            'day_high': day_high,
            'day_low': day_low,
            'day_open': day_open,
            'day_change': ((current_price - day_open) / day_open) * 100 if day_open > 0 else 0
        }
    except Exception as e:
        print(f"Error getting real-time data for {ticker}: {e}")
        return None

def advanced_should_buy(ticker: str, indicators: Dict, current_price: float, realtime_data: Dict) -> Tuple[bool, str]:
    """Improved buy signal detection with better filtering"""
    try:
        # Risk management checks first
        should_skip, skip_reason = should_skip_ticker(ticker)
        if should_skip:
            return False, skip_reason
        
        # Don't buy if already holding
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            return False, "Already holding"
        
        # Market sentiment filter (stricter)
        if memory.market_sentiment == 'BEARISH':
            return False, "Bearish market"
        
        # Signal strength filter (increased threshold)
        strength = memory.signal_strength.get(ticker, 0)
        if strength < STRENGTH_THRESHOLD:
            return False, f"Signal too weak ({strength:.1f})"
        
        # More conservative condition checks
        buy_conditions = []
        reasons = []
        
        # 1. Trend confirmation (more strict)
        sma_20 = safe_extract(indicators.get('sma_20'))
        sma_50 = safe_extract(indicators.get('sma_50'))
        ema_12 = safe_extract(indicators.get('ema_12'))
        ema_26 = safe_extract(indicators.get('ema_26'))
        
        if all([sma_20, sma_50, ema_12, ema_26]):
            # More strict trend requirements
            if (current_price > sma_20 > sma_50 and 
                ema_12 > ema_26 and 
                sma_20 > sma_50 * 1.01):  # At least 1% separation
                buy_conditions.append(True)
                reasons.append("Strong trend")
            else:
                buy_conditions.append(False)
        
        # 2. RSI confirmation (tighter range)
        rsi = safe_extract(indicators.get('rsi_14'))
        if rsi and 30 < rsi < 60:  # More conservative range
            buy_conditions.append(True)
            reasons.append(f"RSI good ({rsi:.1f})")
        else:
            buy_conditions.append(False)
        
        # 3. MACD confirmation (stronger signal required)
        macd = safe_extract(indicators.get('macd'))
        macd_signal = safe_extract(indicators.get('macd_signal'))
        macd_hist = safe_extract(indicators.get('macd_histogram'))
        
        if (macd and macd_signal and macd_hist and 
            macd > macd_signal and macd > 0 and macd_hist > 0):
            buy_conditions.append(True)
            reasons.append("MACD strong")
        else:
            buy_conditions.append(False)
        
        # 4. Volume confirmation (higher threshold)
        volume_sma = safe_extract(indicators.get('volume_sma_10'))
        current_volume = realtime_data.get('volume', 0)
        if volume_sma and current_volume > volume_sma * MIN_VOLUME_SPIKE:
            buy_conditions.append(True)
            reasons.append("High volume")
        else:
            buy_conditions.append(False)
        
        # 5. Volatility filter (stricter)
        volatility_ratio = safe_extract(indicators.get('volatility_ratio'))
        if volatility_ratio and volatility_ratio < 1.5:  # Lower threshold
            buy_conditions.append(True)
            reasons.append("Low volatility")
        else:
            buy_conditions.append(False)
        
        # 6. Price position (not near resistance)
        bb_upper = safe_extract(indicators.get('bb_upper'))
        bb_lower = safe_extract(indicators.get('bb_lower'))
        if bb_lower and bb_upper:
            # Must be in lower 70% of Bollinger Band range
            band_range = bb_upper - bb_lower
            price_position = (current_price - bb_lower) / band_range
            if 0.2 < price_position < 0.7:
                buy_conditions.append(True)
                reasons.append("Good price position")
            else:
                buy_conditions.append(False)
        
        # Need at least 5 out of 6 conditions to be true (stricter)
        conditions_met = sum(buy_conditions)
        if conditions_met >= 4:
            return True, f"Very strong buy ({conditions_met}/6): " + ", ".join(reasons[:3])
        
        return False, f"Insufficient conditions ({conditions_met}/6)"
        
    except Exception as e:
        logger.error(f"Error in improved_should_buy for {ticker}: {e}")
        return False, "Error in analysis"

def advanced_should_sell(ticker: str, indicators: Dict, current_price: float) -> Tuple[bool, str]:
    """Improved sell signal detection with better risk management"""
    if ticker not in memory.holdings or memory.holdings[ticker].get('shares', 0) == 0:
        return False, "No position"
    
    try:
        entry_price = memory.holdings[ticker].get('entry_price', 0)
        entry_time = memory.holdings[ticker].get('entry_time')
        current_pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # Parse entry_time if it's a string
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        # Minimum holding period check
        if entry_time:
            holding_minutes = (datetime.now() - entry_time).total_seconds() / 60
            if holding_minutes < MIN_HOLDING_MINUTES and current_pnl > -5:  # Unless major loss
                return False, f"Min hold time ({MIN_HOLDING_MINUTES - holding_minutes:.0f}min left)"
        
        # Hard stop-loss check (wider than before)
        if ticker in memory.sell_thresholds and current_price <= memory.sell_thresholds[ticker]:
            return True, f"Stop-loss hit (PnL: {current_pnl:+.2f}%)"
        
        # Take profit conditions (higher thresholds)
        if current_pnl > 12:  # Increased from 8%
            rsi = safe_extract(indicators.get('rsi_14'))
            if rsi and rsi > 80:  # Only at extreme overbought
                return True, f"Profit-taking (PnL: {current_pnl:+.2f}%, RSI: {rsi:.1f})"
        
        # Strong trend reversal only
        sma_20 = safe_extract(indicators.get('sma_20'))
        ema_12 = safe_extract(indicators.get('ema_12'))
        ema_26 = safe_extract(indicators.get('ema_26'))
        
        if all([sma_20, ema_12, ema_26]):
            # Only sell on strong reversal signals
            if (current_price < sma_20 * 0.98 and  # 2% below SMA
                ema_12 < ema_26 * 0.99 and         # Clear EMA crossover
                current_pnl < -3):                 # And some loss
                return True, f"Strong reversal (PnL: {current_pnl:+.2f}%)"
        
        # Time-based exit (longer holding period)
        if entry_time:
            holding_time = datetime.now() - entry_time
            if holding_time.days > 10 and current_pnl < 3:  # Increased from 5 days
                return True, f"Time exit (PnL: {current_pnl:+.2f}%, {holding_time.days}d)"
        
        return False, f"Hold (PnL: {current_pnl:+.2f}%)"
        
    except Exception as e:
        logger.error(f"Error in improved_should_sell for {ticker}: {e}")
        return False, "Error in sell analysis"

def execute_advanced_buy(ticker: str, current_price: float, indicators: Dict, reason: str):
    """Execute buy with improved tracking and risk management"""
    try:
        # Record trade time for cooldown
        memory.trade_cooldowns[ticker] = datetime.now()
        
        # Calculate position size (more conservative)
        atr = safe_extract(indicators.get('atr_14'))
        if atr is None or atr <= 0:
            atr = current_price * 0.02
        
        volatility_ratio = safe_extract(indicators.get('volatility_ratio'), 1.0)
        # More conservative sizing
        adjusted_shares = max(1, int(SHARES_TO_BUY / (volatility_ratio * 1.5)))
        
        memory.holdings[ticker] = {
            'shares': adjusted_shares,
            'entry_price': current_price,
            'entry_time': datetime.now()
        }
        
        # More conservative stop-loss (wider)
        support_level = safe_extract(indicators.get('support_1'), current_price * 0.92)
        atr_stop = current_price - (ATR_MULTIPLIER * atr)  # Now 2.5x instead of 2.0x
        dynamic_stop = max(support_level * 0.95, atr_stop)  # 5% buffer below support
        
        memory.sell_thresholds[ticker] = dynamic_stop
        memory.highest_prices[ticker] = current_price
        
        memory.total_trades += 1
        
        # Save memory after trade
        # memory.save_memory()
        
        # Send buy alert
        symbol = ticker.replace('.NS', '').replace('.BO', '')
        signal_strength = memory.signal_strength.get(ticker, 0)
        
        message = f"🟢 *IMPROVED BUY SIGNAL*\n"
        message += f"📈 {symbol} - Rs.{current_price:.2f}\n"
        message += f"💰 Shares: {adjusted_shares} (Conservative sizing)\n"
        message += f"🎯 Signal Strength: {signal_strength:.1f}/100\n"
        message += f"🛑 Stop-loss: Rs.{dynamic_stop:.2f}\n"
        message += f"💡 Reason: {reason}"
        
        send_telegram_message(message)
        logger.info(f"[IMPROVED BUY] {symbol} @ Rs.{current_price:.2f} | Strength: {signal_strength:.1f}")
        
    except Exception as e:
        logger.error(f"Error executing improved buy for {ticker}: {e}")

def execute_advanced_sell(ticker: str, current_price: float, reason: str):
    """Execute sell with improved tracking"""
    try:
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
        total_change = (current_price - entry_price) * shares
        change_percent = ((current_price - entry_price) / entry_price) * 100
        holding_period = datetime.now() - entry_time
        
        # Update consecutive loss counter
        if total_change < 0:
            memory.consecutive_losses[ticker] = memory.consecutive_losses.get(ticker, 0) + 1
        else:
            memory.consecutive_losses[ticker] = 0  # Reset on profit
            memory.profitable_trades += 1
        
        # Record trade time for cooldown
        memory.trade_cooldowns[ticker] = datetime.now()
        
        # Update session statistics
        memory.total_pnl += total_change
        
        # Clear position
        memory.holdings[ticker] = {'shares': 0, 'entry_price': 0}
        if ticker in memory.sell_thresholds:
            del memory.sell_thresholds[ticker]
        if ticker in memory.highest_prices:
            del memory.highest_prices[ticker]
        
        # Reset alerts
        memory.alerts_sent[ticker] = {'52w_high': False, 'breakout': False, 'support': False}
        
        # Save memory after trade
        # memory.save_memory()
        
        # Send sell alert
        symbol = ticker.replace('.NS', '').replace('.BO', '')
        profit_emoji = "✅" if total_change >= 0 else "❌"
        
        message = f"🔴 *IMPROVED SELL SIGNAL*\n"
        message += f"📉 {symbol} - Rs.{current_price:.2f}\n"
        message += f"💼 Sold {shares} shares\n"
        message += f"{profit_emoji} P&L: Rs.{total_change:.2f} ({change_percent:+.2f}%)\n"
        message += f"⏱️ Held: {holding_period.days}d {holding_period.seconds//3600}h\n"
        message += f"💡 Reason: {reason}"
        
        send_telegram_message(message)
        logger.info(f"[IMPROVED SELL] {symbol} @ Rs.{current_price:.2f} | P&L: Rs.{total_change:.2f}")
        
    except Exception as e:
        logger.error(f"Error executing improved sell for {ticker}: {e}")


def update_dynamic_trailing_stop(ticker: str, current_price: float, indicators: Dict):
    """Update dynamic trailing stop-loss"""
    if ticker not in memory.holdings or memory.holdings[ticker].get('shares', 0) == 0:
        return
    
    try:
        atr = safe_extract(indicators.get('atr_14'))
        if atr is None or atr <= 0:
            atr = current_price * 0.02
        
        # Update highest price
        if ticker not in memory.highest_prices:
            memory.highest_prices[ticker] = current_price
        else:
            memory.highest_prices[ticker] = max(memory.highest_prices[ticker], current_price)
        
        # Dynamic ATR multiplier based on profit
        entry_price = memory.holdings[ticker].get('entry_price', current_price)
        profit_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # Tighten stop-loss as profit increases
        if profit_percent > 10:
            atr_multiplier = 1.5  # Tighter stop for high profits
        elif profit_percent > 5:
            atr_multiplier = 1.8  # Medium stop
        else:
            atr_multiplier = ATR_MULTIPLIER  # Standard stop
        
        # Calculate new trailing stop
        new_stop = memory.highest_prices[ticker] - (atr_multiplier * atr)
        
        # Support level consideration
        support_level = safe_extract(indicators.get('support_1'))
        if support_level and support_level < new_stop:
            new_stop = max(new_stop, support_level * 0.98)  # Small buffer below support
        
        # Only update if new stop is higher (trailing up)
        if ticker not in memory.sell_thresholds:
            memory.sell_thresholds[ticker] = new_stop
        else:
            memory.sell_thresholds[ticker] = max(memory.sell_thresholds[ticker], new_stop)
            
    except Exception as e:
        print(f"Error updating trailing stop for {ticker}: {e}")

def check_advanced_alerts(ticker: str, current_price: float, indicators: Dict, realtime_data: Dict):
    """Check for various advanced alerts"""
    try:
        symbol = ticker.replace('.NS', '').replace('.BO', '')
        
        # 1. Breakout Alert
        if not memory.alerts_sent[ticker]['breakout']:
            bb_upper = safe_extract(indicators.get('bb_upper'))
            day_high = realtime_data.get('day_high', 0)
            volume_sma = safe_extract(indicators.get('volume_sma_10'))
            current_volume = realtime_data.get('volume', 0)
            
            if all([bb_upper, volume_sma]) and current_price > bb_upper and current_volume > volume_sma * 2:
                # message = f" *BREAKOUT ALERT*\n"
                # message += f" {symbol} broke above Bollinger Band\n"
                # message += f" Price: Rs.{current_price:.2f} (vs Rs.{bb_upper:.2f})\n"
                # message += f" Volume spike: {(current_volume/volume_sma):.1f}x average\n"
                # message += f" Strong momentum detected!"
                
                message = f"🚨 *BREAKOUT ALERT* 🚨\n"
                message += f"📈 {symbol} broke above Bollinger Band\n"
                message += f"💰 Price: Rs.{current_price:.2f} (vs Rs.{bb_upper:.2f})\n"
                message += f"📊 Volume spike: {(current_volume/volume_sma):.1f}x average\n"
                message += f"🔥 Strong momentum detected! 🚀"

                send_telegram_message(message)
                memory.alerts_sent[ticker]['breakout'] = True
        
        # 2. Support Level Alert (for holdings)
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            if not memory.alerts_sent[ticker]['support']:
                support_level = safe_extract(indicators.get('support_1'))
                if support_level and current_price < support_level * 1.02:  # Within 2% of support
                    # message = f" *SUPPORT LEVEL ALERT*\n"
                    # message += f" {symbol} near support at Rs.{support_level:.2f}\n"
                    # message += f" Current: Rs.{current_price:.2f}\n"
                    # message += f" Key level to watch!"

                    # message = f"⚠️ *SUPPORT LEVEL ALERT* ⚠️\n"
                    # message += f"📉 {symbol} near support at Rs.{support_level:.2f}\n"
                    # message += f"💵 Current: Rs.{current_price:.2f}\n"
                    # message += f"👀 Key level to watch! 🎯"
                    
                    # send_telegram_message(message)
                    memory.alerts_sent[ticker]['support'] = True
        
        # 3. Enhanced 52-week high alert
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            if not memory.alerts_sent[ticker]['52w_high']:
                high_52w = indicators.get('52w_high', 0)
                distance_from_high = safe_extract(indicators.get('distance_from_52w_high'))
                
                if distance_from_high and distance_from_high < 2:  # Within 2% of 52w high
                    entry_price = memory.holdings[ticker].get('entry_price', 0)
                    profit = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    
                    # message = f" *52-WEEK HIGH ALERT*\n"
                    # message += f" {symbol} near 52W high!\n"
                    # message += f" Current: Rs.{current_price:.2f}\n"
                    # message += f" 52W High: Rs.{high_52w:.2f}\n"
                    # message += f" Your profit: {profit:+.2f}%\n"
                    # message += f" Consider exit strategy"
                                        
                    message = f"🏆 *52-WEEK HIGH ALERT* 🏆\n"
                    message += f"🚀 {symbol} near 52W high!\n"
                    message += f"💰 Current: Rs.{current_price:.2f}\n"
                    message += f"📊 52W High: Rs.{high_52w:.2f}\n"
                    message += f"💸 Your profit: {profit:+.2f}%\n"
                    message += f"🤔 Consider exit strategy 📋"

                    send_telegram_message(message)
                    memory.alerts_sent[ticker]['52w_high'] = True
                    
    except Exception as e:
        print(f"Error checking alerts for {ticker}: {e}")

def has_earnings_soon(ticker: str) -> bool:
    """Check if stock has earnings in next 2 days"""
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

# ============================
# MAIN ANALYSIS FUNCTION
# ============================

# def analyze_stock_advanced(ticker: str):
#     """Advanced stock analysis with comprehensive signal generation"""
#     try:
#         # Check shutdown flag        
#         if memory.shutdown_flag:
#             return
        
#         # Fetch enhanced historical data
#         historical_df = get_stock_data(ticker, period="6mo")
#         if historical_df is None or historical_df.empty:
#             print(f"No historical data for {ticker}")
#             return
        
#         # Calculate advanced indicators
#         indicators = calculate_advanced_indicators(historical_df)
#         if not indicators:
#             print(f"Failed to calculate indicators for {ticker}")
#             return
        
#         # Get enhanced real-time data
#         realtime_data = get_realtime_data(ticker)
#         if not realtime_data:
#             print(f"No real-time data for {ticker}")
#             return
        
#         current_price = realtime_data['price']
        
#         # Calculate signal strength
#         signal_strength = calculate_signal_strength(ticker, indicators, current_price)
#         memory.signal_strength[ticker] = signal_strength
        
#         # Update trailing stops for existing positions
#         if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#             update_dynamic_trailing_stop(ticker, current_price, indicators)
        
#         # Check for advanced alerts
#         check_advanced_alerts(ticker, current_price, indicators, realtime_data)
        
#         # Advanced trading decisions
#         should_sell, sell_reason = advanced_should_sell(ticker, indicators, current_price)
#         if should_sell:
#             execute_advanced_sell(ticker, current_price, sell_reason)
#             memory.last_action_status[ticker] = 'SELL_SIGNAL'
#         else:
#             should_buy, buy_reason = advanced_should_buy(ticker, indicators, current_price, realtime_data)
#             if should_buy:
#                 execute_advanced_buy(ticker, current_price, indicators, buy_reason)
#                 memory.last_action_status[ticker] = 'BUY_SIGNAL'
#             else:
#                 # Determine current status
#                 if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#                     memory.last_action_status[ticker] = f'HOLD ({sell_reason})'
#                 else:
#                     memory.last_action_status[ticker] = f'WAIT ({buy_reason})'
                    
#     except Exception as e:
#         print(f"Error in advanced analysis for {ticker}: {e}")
#         memory.last_action_status[ticker] = 'ERROR'

# ============================
# STAGED ANALYSIS FUNCTION
# ============================

# def analyze_stock_staged(ticker: str):
#     """Staged stock analysis with memory optimization"""
#     try:
#         # Check shutdown flag        
#         if memory.shutdown_flag:
#             return
        
#         # Pre-analysis memory check
#         current_process_mb = psutil.Process().memory_info().rss / 1024 / 1024
#         if current_process_mb > 400:  # 400MB threshold
#             logger.warning(f"High memory usage before {ticker} analysis: {current_process_mb:.1f}MB")
#             cleanup_memory()
        
#         # Stage 1: Fetch historical data (will use cache if available)
#         logger.debug(f"Stage 1: Loading historical data for {ticker}")
#         historical_df = data_cache.get_cached_historical(ticker, period="6mo")
#         if historical_df is None or historical_df.empty:
#             logger.warning(f"No historical data for {ticker}")
#             return
        
#         # Stage 2: Calculate indicators (will use cache if available)  
#         logger.debug(f"Stage 2: Calculating indicators for {ticker}")
#         indicators = data_cache.get_cached_indicators(ticker, historical_df)
#         if not indicators:
#             logger.warning(f"Failed to calculate indicators for {ticker}")
#             return
        
#         # Stage 3: Get real-time data (lightweight)
#         logger.debug(f"Stage 3: Getting real-time data for {ticker}")
#         realtime_data = get_realtime_data(ticker)
#         if not realtime_data:
#             logger.warning(f"No real-time data for {ticker}")
#             return
        
#         current_price = realtime_data['price']
        
#         # Stage 4: Quick analysis using cached indicators
#         logger.debug(f"Stage 4: Performing analysis for {ticker}")
        
#         # Calculate signal strength using cached values
#         signal_strength = calculate_signal_strength(ticker, indicators, current_price)
#         memory.signal_strength[ticker] = signal_strength
        
#         # Update trailing stops for existing positions
#         if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#             update_dynamic_trailing_stop(ticker, current_price, indicators)
        
#         # Check for alerts
#         check_advanced_alerts(ticker, current_price, indicators, realtime_data)
        
#         # Trading decisions
#         should_sell, sell_reason = advanced_should_sell(ticker, indicators, current_price)
#         if should_sell:
#             execute_advanced_sell(ticker, current_price, sell_reason)
#             memory.last_action_status[ticker] = 'SELL_SIGNAL'
#         else:
#             should_buy, buy_reason = advanced_should_buy(ticker, indicators, current_price, realtime_data)
#             if should_buy:
#                 execute_advanced_buy(ticker, current_price, indicators, buy_reason)
#                 memory.last_action_status[ticker] = 'BUY_SIGNAL'
#             else:
#                 if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#                     memory.last_action_status[ticker] = f'HOLD ({sell_reason})'
#                 else:
#                     memory.last_action_status[ticker] = f'WAIT ({buy_reason})'
        
#         # Stage 5: Memory cleanup after analysis
#         logger.debug(f"Stage 5: Post-analysis cleanup for {ticker}")
        
#         # Force garbage collection every 10 tickers
#         import gc
#         if TICKERS.index(ticker) % 10 == 9:
#             gc.collect()
            
#         # Log memory usage
#         post_analysis_mb = psutil.Process().memory_info().rss / 1024 / 1024
#         logger.debug(f"Memory after {ticker}: {post_analysis_mb:.1f}MB")
                    
#     except Exception as e:
#         logger.error(f"Error in staged analysis for {ticker}: {e}")
#         memory.last_action_status[ticker] = 'ERROR'

# def analyze_stock_staged(ticker: str):
#     """Fixed staged analysis with proper caching"""
#     try:
#         if memory.shutdown_flag:
#             return
        
#         # Memory check
#         current_process_mb = psutil.Process().memory_info().rss / 1024 / 1024
#         if current_process_mb > 350:  # Lower threshold for safety
#             logger.warning(f"High memory before {ticker}: {current_process_mb:.1f}MB")
#             enhanced_cleanup_memory()
        
#         # Stage 1: Get cached historical data
#         historical_df = data_cache.get_cached_historical(ticker, period="6mo")
#         if historical_df is None or historical_df.empty:
#             logger.warning(f"No historical data for {ticker}")
#             return
        
#         # Stage 2: Get cached compressed indicators
#         indicators = data_cache.get_cached_indicators(ticker, historical_df)
#         if not indicators:
#             logger.warning(f"Failed to calculate indicators for {ticker}")
#             return
        
#         # Stage 3: Get real-time data
#         realtime_data = get_realtime_data(ticker)
#         if not realtime_data:
#             logger.warning(f"No real-time data for {ticker}")
#             return
        
#         current_price = realtime_data['price']
        
#         # Stage 4: Use cached signal strength calculation
#         signal_strength = calculate_signal_strength(ticker, indicators, current_price)
#         memory.signal_strength[ticker] = signal_strength
        
#         # Rest of analysis using cached indicators...
#         # (Your existing trading logic here)
#         # Check shutdown flag        
#         if memory.shutdown_flag:
#             return
        
#         # Fetch enhanced historical data
#         historical_df = get_stock_data(ticker, period="6mo")
#         if historical_df is None or historical_df.empty:
#             print(f"No historical data for {ticker}")
#             return
        
#         # Calculate advanced indicators
#         indicators = calculate_advanced_indicators(historical_df)
#         if not indicators:
#             print(f"Failed to calculate indicators for {ticker}")
#             return
        
#         # Get enhanced real-time data
#         realtime_data = get_realtime_data(ticker)
#         if not realtime_data:
#             print(f"No real-time data for {ticker}")
#             return
        
#         current_price = realtime_data['price']
        
#         # Calculate signal strength
#         signal_strength = calculate_signal_strength(ticker, indicators, current_price)
#         memory.signal_strength[ticker] = signal_strength
        
#         # Update trailing stops for existing positions
#         if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#             update_dynamic_trailing_stop(ticker, current_price, indicators)
        
#         # Check for advanced alerts
#         check_advanced_alerts(ticker, current_price, indicators, realtime_data)
        
#         # Advanced trading decisions
#         should_sell, sell_reason = advanced_should_sell(ticker, indicators, current_price)
#         if should_sell:
#             execute_advanced_sell(ticker, current_price, sell_reason)
#             memory.last_action_status[ticker] = 'SELL_SIGNAL'
#         else:
#             should_buy, buy_reason = advanced_should_buy(ticker, indicators, current_price, realtime_data)
#             if should_buy:
#                 execute_advanced_buy(ticker, current_price, indicators, buy_reason)
#                 memory.last_action_status[ticker] = 'BUY_SIGNAL'
#             else:
#                 # Determine current status
#                 if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#                     memory.last_action_status[ticker] = f'HOLD ({sell_reason})'
#                 else:
#                     memory.last_action_status[ticker] = f'WAIT ({buy_reason})'

#     except Exception as e:
#         logger.error(f"Error in fixed staged analysis for {ticker}: {e}")
#         memory.last_action_status[ticker] = 'ERROR'

def analyze_stock_staged(ticker: str):
    """Fixed staged analysis - CLEAN VERSION"""
    try:
        if memory.shutdown_flag:
            return
        
        # Memory check
        current_process_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if current_process_mb > 350:
            logger.warning(f"High memory before {ticker}: {current_process_mb:.1f}MB")
            enhanced_cleanup_memory()
        
        # Stage 1: Get cached historical data
        historical_df = data_cache.get_cached_historical(ticker, period="6mo")
        if historical_df is None or historical_df.empty:
            logger.warning(f"No historical data for {ticker}")
            return
        
        # Stage 2: Get cached compressed indicators (single values only)
        indicators = data_cache.get_cached_indicators(ticker, historical_df)
        if not indicators:
            logger.warning(f"Failed to calculate indicators for {ticker}")
            return
        
        # Stage 3: Get real-time data
        realtime_data = get_realtime_data(ticker)
        if not realtime_data:
            logger.warning(f"No real-time data for {ticker}")
            return
        
        current_price = realtime_data['price']
        
        # Stage 4: Analysis using compressed indicators
        signal_strength = calculate_signal_strength(ticker, indicators, current_price)
        memory.signal_strength[ticker] = signal_strength
        
        # Update trailing stops for existing positions
        if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
            update_dynamic_trailing_stop(ticker, current_price, indicators)
        
        # Check for alerts
        check_advanced_alerts(ticker, current_price, indicators, realtime_data)
        
        # Trading decisions
        should_sell, sell_reason = advanced_should_sell(ticker, indicators, current_price)
        if should_sell:
            execute_advanced_sell(ticker, current_price, sell_reason)
            memory.last_action_status[ticker] = 'SELL_SIGNAL'
        else:
            should_buy, buy_reason = advanced_should_buy(ticker, indicators, current_price, realtime_data)
            if should_buy:
                execute_advanced_buy(ticker, current_price, indicators, buy_reason)
                memory.last_action_status[ticker] = 'BUY_SIGNAL'
            else:
                if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
                    memory.last_action_status[ticker] = f'HOLD ({sell_reason})'
                else:
                    memory.last_action_status[ticker] = f'WAIT ({buy_reason})'

    except Exception as e:
        logger.error(f"Error in analysis for {ticker}: {e}")
        memory.last_action_status[ticker] = 'ERROR'


# def analyze_stock_advanced(ticker: str):
#     """Advanced stock analysis with comprehensive signal generation"""
#     try:
#         # Check shutdown flag        
#         if memory.shutdown_flag:
#             return
        
#         # Fetch enhanced historical data
#         historical_df = get_stock_data(ticker, period="6mo")
#         if historical_df is None or historical_df.empty:
#             print(f"No historical data for {ticker}")
#             return
        
#         # Calculate advanced indicators
#         indicators = calculate_advanced_indicators(historical_df)
#         if not indicators:
#             print(f"Failed to calculate indicators for {ticker}")
#             return
        
#         # Get enhanced real-time data
#         realtime_data = get_realtime_data(ticker)
#         if not realtime_data:
#             print(f"No real-time data for {ticker}")
#             return
        
#         current_price = realtime_data['price']
        
#         # Calculate signal strength
#         signal_strength = calculate_signal_strength(ticker, indicators, current_price)
#         memory.signal_strength[ticker] = signal_strength
        
#         # Update trailing stops for existing positions
#         if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#             update_dynamic_trailing_stop(ticker, current_price, indicators)
        
#         # Check for advanced alerts
#         check_advanced_alerts(ticker, current_price, indicators, realtime_data)
        
#         # Advanced trading decisions
#         should_sell, sell_reason = advanced_should_sell(ticker, indicators, current_price)
#         if should_sell:
#             execute_advanced_sell(ticker, current_price, sell_reason)
#             memory.last_action_status[ticker] = 'SELL_SIGNAL'
#         else:
#             should_buy, buy_reason = advanced_should_buy(ticker, indicators, current_price, realtime_data)
#             if should_buy:
#                 execute_advanced_buy(ticker, current_price, indicators, buy_reason)
#                 memory.last_action_status[ticker] = 'BUY_SIGNAL'
#             else:
#                 # Determine current status
#                 if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
#                     memory.last_action_status[ticker] = f'HOLD ({sell_reason})'
#                 else:
#                     memory.last_action_status[ticker] = f'WAIT ({buy_reason})'
                    
#     except Exception as e:
#         print(f"Error in advanced analysis for {ticker}: {e}")
#         memory.last_action_status[ticker] = 'ERROR'

# ============================
# ENHANCED CONSOLE OUTPUT
# ============================

def print_advanced_status_table():
    """Print comprehensive advanced status table"""
    table_data = []
    
    for ticker in TICKERS:
        try:
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            
            realtime_data = get_realtime_data(ticker)
            current_price = realtime_data['price'] if realtime_data else 0.0
            day_change = realtime_data.get('day_change', 0.0) if realtime_data else 0.0
            
            historical_df = get_stock_data(ticker, period="6mo")
            if historical_df is not None and not historical_df.empty:
                indicators = calculate_advanced_indicators(historical_df)
            else:
                indicators = {}
            
            # Key indicators
            rsi = safe_extract(indicators.get('rsi_14'), 0.0)
            macd = safe_extract(indicators.get('macd'), 0.0)
            atr = safe_extract(indicators.get('atr_14'), 0.0)
            signal_strength = memory.signal_strength.get(ticker, 0.0)
            
            # Position details
            status = memory.last_action_status.get(ticker, 'WAIT')
            entry_price = 0.0
            sell_threshold = 0.0
            change_percent = 0.0
            position_value = 0.0
            
            if ticker in memory.holdings and memory.holdings[ticker].get('shares', 0) > 0:
                shares = memory.holdings[ticker]['shares']
                entry_price = memory.holdings[ticker]['entry_price']
                sell_threshold = memory.sell_thresholds.get(ticker, 0.0)
                position_value = shares * current_price
                if entry_price > 0:
                    change_percent = ((current_price - entry_price) / entry_price) * 100
            
            # Format strings
            current_price_str = f"Rs.{current_price:.2f}" if current_price > 0 else "N/A"
            day_change_str = f"{day_change:+.2f}%" if day_change != 0 else "0.00%"
            entry_price_str = f"Rs.{entry_price:.2f}" if entry_price > 0 else "--"
            rsi_str = f"{rsi:.1f}" if rsi > 0 else "N/A"
            macd_str = f"{macd:.3f}" if macd != 0 else "N/A"
            atr_str = f"{atr:.2f}" if atr > 0 else "N/A"
            signal_strength_str = f"{signal_strength:.0f}" if signal_strength > 0 else "0"
            sell_threshold_str = f"Rs.{sell_threshold:.2f}" if sell_threshold > 0 else "--"
            change_percent_str = f"{change_percent:+.2f}%" if change_percent != 0 else "--"
            position_value_str = f"Rs.{position_value:.0f}" if position_value > 0 else "--"
            
            # Color coding for signal strength
            if signal_strength >= 75:
                signal_strength_str = f"🟢{signal_strength_str}"
            elif signal_strength >= 50:
                signal_strength_str = f"🟡{signal_strength_str}"
            elif signal_strength > 0:
                signal_strength_str = f"🔴{signal_strength_str}"
            
            # Truncate status for display
            display_status = status[:15] + "..." if len(status) > 15 else status
            
            table_data.append([
                symbol,
                current_price_str,
                day_change_str,
                entry_price_str,
                rsi_str,
                macd_str,
                atr_str,
                signal_strength_str,
                sell_threshold_str,
                change_percent_str,
                position_value_str,
                display_status
            ])
            
        except Exception as e:
            table_data.append([
                symbol, "ERROR", "N/A", "--", "--", "--", "--", "0", "--", "--", "--", "ERROR"
            ])
            print(f"Error processing {ticker}: {e}")
    
    # Calculate summary statistics
    total_positions = len([row for row in table_data if "Rs." in row[10] and row[10] != "--"])
    waiting_positions = len(TICKERS) - total_positions
    strong_signals = len([row for row in table_data if "🟢" in row[7]])
    weak_signals = len([row for row in table_data if "🔴" in row[7]])
    
    print("\n" + "="*150)
    print("ADVANCED STOCK TRADING BOT - COMPREHENSIVE STATUS")
    print("="*150)
    print(tabulate(table_data, headers=[
        "Ticker", "Price", "Day%", "Entry", "RSI", "MACD", "ATR", 
        "Signal", "Stop", "P&L%", "Value", "Status"
    ], tablefmt="grid"))
    print("="*150)
    
    print(f"POSITIONS: {total_positions} ACTIVE | {waiting_positions} WAITING")
    print(f"SIGNALS: {strong_signals} STRONG 🟢 | {weak_signals} WEAK 🔴")
    print(f"MARKET SENTIMENT: {memory.market_sentiment}")
    print(f"SESSION P&L: Rs.{memory.total_pnl:.2f} | MAX DRAWDOWN: {memory.max_drawdown:.2f}%")
    print(f"WIN RATE: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "WIN RATE: 0%")
    print(f"LAST UPDATED: {datetime.now().strftime('%H:%M:%S')}")
    print("="*150)

# ============================
# TIME MANAGEMENT
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
# MAIN TRADING LOOP
# ============================

# def main_advanced_trading_loop():
#     """Main advanced trading loop with enhanced features"""
#     print(" Advanced Stock Trading Bot Started!")
#     # send_telegram_message(" *Advanced Stock Trading Bot v2.0 Started!*\n Enhanced with 15+ technical indicators\n Smart signal strength analysis\n Dynamic position sizing & stop-loss")
#     send_telegram_message("🤖 *Advanced Stock Trading Bot v2.0 Started!* 🚀\n"
#                         "⚡ Enhanced with 15+ technical indicators\n"
#                         "🧠 Smart signal strength analysis\n"
#                         "🎯 Dynamic position sizing & stop-loss ✅")

#     loop_count = 0
#     last_cleanup = datetime.now()

#     while True:
#         try:
#             # Check shutdown flag
#             if memory.shutdown_flag:
#                 break

#             current_time = datetime.now()
#             loop_count += 1
            
#             if (current_time - last_cleanup).total_seconds() > 7200:  # 2 hours
#                 logger.info("Starting scheduled memory cleanup...")
#                 cleanup_memory()
#                 last_cleanup = current_time

#             # Send alive notifications
#             if is_alive_check_time():
#                 if (memory.last_alive_check is None or 
#                     (current_time - memory.last_alive_check).total_seconds() > 3600):
#                     send_alive_notification()
            
#             # Only trade during market hours
#             if not is_market_hours():
#                 print(f"[{current_time.strftime('%H:%M:%S')}] Market closed. Waiting...")
#                 time.sleep(CHECK_INTERVAL)
#                 continue
            
#             print(f"\n[{current_time.strftime('%H:%M:%S')}] Advanced Analysis Cycle #{loop_count}")

#             if loop_count % 6 == 0:
#                 monitor_system_health()

#             # Analyze all stocks with advanced logic
#             for i, ticker in enumerate(TICKERS):
#                 if memory.shutdown_flag:
#                     break
#                 print(f"Analyzing {ticker} ({i+1}/{len(TICKERS)})...", end=" ")
#                 analyze_stock_advanced(ticker)
#                 print("✓")
#                 time.sleep(2)  # Rate limiting

#             # Check shutdown flag after analysis
#             if memory.shutdown_flag:
#                 break

#             # Update market sentiment
#             memory.market_sentiment = calculate_market_sentiment()
            
#             # Print detailed status table
#             if loop_count % 3 == 1 or is_alive_check_time():  # Every 3 cycles or during alive check times
#                 print_advanced_status_table()
            
#             # Send periodic summary to Telegram
#             if loop_count % 12 == 0:  # Every hour during market
#                 active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
#                 strong_signals = sum(1 for strength in memory.signal_strength.values() if strength > 70)
                
#                 # summary_msg = f" *Hourly Summary*\n"
#                 # summary_msg += f" Active: {active_positions} | Strong signals: {strong_signals}\n"
#                 # summary_msg += f" Market: {memory.market_sentiment} | P&L: Rs.{memory.total_pnl:.2f}\n"
#                 # summary_msg += f" Win rate: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "🎯 Win rate: 0%"
                
#                 summary_msg = f"⏰ *Hourly Summary* 📊\n"
#                 summary_msg += f"🔄 Active: {active_positions} | ⚡ Strong signals: {strong_signals}\n"
#                 summary_msg += f"📈 Market: {memory.market_sentiment} | 💰 P&L: Rs.{memory.total_pnl:.2f}\n"
#                 summary_msg += f"🎯 Win rate: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "🎯 Win rate: 0%"

#                 send_telegram_message(summary_msg)
            
#             print(f"[{current_time.strftime('%H:%M:%S')}] Cycle complete. Next analysis in {CHECK_INTERVAL//60} minutes...")
            
#         except KeyboardInterrupt:
#             print("\n🛑 Advanced Bot stopped by user")
#             cleanup_and_exit()
#             break
#         except Exception as e:
#             print(f"Error in main advanced loop: {e}")
#             # error_msg = f" *Advanced Bot Error*\nCycle #{loop_count}\nError: {str(e)[:100]}\nBot continuing..."
#             error_msg = f"🚨 *Advanced Bot Error* ⚠️\n🔄 Cycle #{loop_count}\n❌ Error: {str(e)[:100]}\n✅ Bot continuing..."
#             send_telegram_message(error_msg)
        
#         time.sleep(CHECK_INTERVAL)
# ============================
# MODIFIED MAIN LOOP
# ============================

def main_advanced_trading_loop():
    """Main trading loop with staged loading"""
    logger.info("🤖 Advanced Stock Trading Bot v2.0 with Loading Started! 🚀")
    send_telegram_message("🤖 *Advanced Stock Trading Bot v2.0 Started!* 🚀\n"
                        "⚡ Enhanced with data loading\n"
                        "🧠 Memory-optimized analysis\n"
                        "🎯 40 ticker monitoring ✅")

    loop_count = 0
    last_cleanup = datetime.now()

    while True:
        try:
            if memory.shutdown_flag:
                break

            current_time = datetime.now()
            loop_count += 1
            
            # Memory management
            if (current_time - last_cleanup).total_seconds() > 7200:  # 2 hours
                logger.info("Starting scheduled memory cleanup...")
                cleanup_memory()
                data_cache.cleanup_old_cache_entries()
                last_cleanup = current_time

            # Alive notifications
            if is_alive_check_time():
                if (memory.last_alive_check is None or 
                    (current_time - memory.last_alive_check).total_seconds() > 3600):
                    send_alive_notification()
            
            # Market hours check
            if not is_market_hours():
                logger.info(f"[{current_time.strftime('%H:%M:%S')}] Market closed. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            logger.info(f"\n[{current_time.strftime('%H:%M:%S')}] Analysis Cycle #{loop_count}")

            # System health monitoring
            if loop_count % 6 == 0:
                monitor_staged_system_health()

            # Log cache status
            cache_size = data_cache.get_cache_size_mb()
            process_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory status: Process={process_memory:.1f}MB, Cache={cache_size:.1f}MB")

            # Analyze all stocks with staged loading
            for i, ticker in enumerate(TICKERS):
                if memory.shutdown_flag:
                    break
                
                logger.debug(f"Analyzing {ticker} ({i+1}/{len(TICKERS)})...")
                analyze_stock_staged(ticker)  # Use staged analysis
                logger.debug(f"{ticker} ✓")
                
                # Brief pause between tickers
                time.sleep(1)

            if memory.shutdown_flag:
                break

            # Update market sentiment
            memory.market_sentiment = calculate_market_sentiment()
            
            # Status reporting
            if loop_count % 3 == 1 or is_alive_check_time():
                print_advanced_status_table()
            
            # Hourly summary
            if loop_count % 12 == 0:
                active_positions = sum(1 for ticker in memory.holdings if memory.holdings[ticker].get('shares', 0) > 0)
                strong_signals = sum(1 for strength in memory.signal_strength.values() if strength > 70)
                
                summary_msg = f"⏰ *Hourly Summary* 📊\n"
                summary_msg += f"🔥 Active: {active_positions} | ⚡ Strong signals: {strong_signals}\n"
                summary_msg += f"📈 Market: {memory.market_sentiment} | 💰 P&L: Rs.{memory.total_pnl:.2f}\n"
                summary_msg += f"🎯 Win rate: {(memory.profitable_trades/memory.total_trades*100):.1f}%" if memory.total_trades > 0 else "🎯 Win rate: 0%"
                summary_msg += f"\n💾 Memory: {process_memory:.1f}MB"

                send_telegram_message(summary_msg)
            
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] Cycle complete. Next analysis in {CHECK_INTERVAL//60} minutes...")
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user")
            cleanup_and_exit()
            break
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            error_msg = f"🚨 *Bot Error* ⚠️\n🔄 Cycle #{loop_count}\n❌ Error: {str(e)[:100]}\n✅ Bot continuing..."
            send_telegram_message(error_msg)
        
        time.sleep(CHECK_INTERVAL)

def main():
    """Main entry point for the advanced trading bot"""
    # Set up exit handlers first
    setup_exit_handlers()
    setup_enhanced_diagnostics(send_telegram_message)

    # Verify required libraries
    try:
        import talib
        from tabulate import tabulate
        logger.info("All required libraries verified")
    except ImportError as e:
        if 'talib' in str(e):
            logger.error(" TA-Lib not installed. Install with: pip install TA-Lib")
            logger.error("On Windows, you might need to download the wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        elif 'tabulate' in str(e):
            logger.error(" tabulate not installed. Install with: pip install tabulate")
        sys.exit(1)
    
    # Configuration check
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logger.warning(" WARNING: Telegram bot token not configured. Messages will print to console.")
    
    # Log advanced configuration settings
    logger.info(" Advanced Trading Bot Configuration:")
    logger.info(f"    Monitoring {len(TICKERS)} stocks")
    logger.info(f"    Check interval: {CHECK_INTERVAL//60} minutes")
    logger.info(f"    ATR Multiplier: {ATR_MULTIPLIER}")
    logger.info(f"    RSI Range: {RSI_OVERSOLD}-{RSI_OVERBOUGHT}")
    logger.info(f"    Volume Spike Threshold: {MIN_VOLUME_SPIKE}x")
    logger.info(f"    Signal Strength Threshold: {STRENGTH_THRESHOLD}")
    
    # Print initial advanced status table
    logger.info(" Initializing advanced stock analysis...")
    try:
        print_advanced_status_table()
    except Exception as e:
        logger.error(f"Error in initial analysis: {e}")
    
    # Start the advanced trading bot
    try:
        main_advanced_trading_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        cleanup_and_exit()
    finally:
        print_final_summary()

if __name__ == "__main__":
    main()
    setup_enhanced_diagnostics()
    logger.info("Diagnostics test - press Ctrl+C to test shutdown detection")
    try:
        while True:
            time.sleep(10)
            monitor_staged_system_health()
    except KeyboardInterrupt:
        logger.info("Manual interrupt received")
