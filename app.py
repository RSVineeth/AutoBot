from flask import Flask, jsonify, Response
import os
import threading
from trading_loop import main, memory  # Import memory for status
import time
from datetime import datetime

varport = int(os.getenv("PORT", 10000))
app = Flask(__name__)

bot_status = {
    'started': False,
    'start_time': None,
    'thread_alive': False
}

@app.route('/', methods=['GET', 'HEAD'])
def home():
    # return "Trading Bot is running!", 200
    return Response("Bot is running!", status=200)

@app.route('/health', methods=['GET'])
def health():
    """Health check for Render"""
    try:
        active_positions = sum(
            1 for t in memory.holdings 
            if memory.holdings.get(t, {}).get('shares', 0) > 0
        )
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'bot_started': bot_status['started'],
            'thread_alive': bot_status['thread_alive'],
            'active_positions': active_positions,
            'total_trades': memory.total_trades,
            'session_pnl': float(memory.total_pnl)
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/uptime', methods=['GET'])
def uptime():
    if bot_status['start_time']:
        uptime_seconds = (datetime.now() - bot_status['start_time']).total_seconds()
        uptime_str = f"{int(uptime_seconds//3600)}h {int((uptime_seconds%3600)//60)}m"
    else:
        uptime_str = "Not started"
    
    return jsonify({
        'uptime': uptime_str,
        'current_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'thread_alive': bot_status['thread_alive']
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Detailed status"""
    try:
        holdings = {
            ticker: {
                'shares': data.get('shares', 0),
                'entry_price': data.get('entry_price', 0)
            }
            for ticker, data in memory.holdings.items()
            if data.get('shares', 0) > 0
        }
        
        return jsonify({
            'bot_status': bot_status,
            'active_holdings': holdings,
            'market_sentiment': memory.market_sentiment,
            'profitable_trades': memory.profitable_trades,
            'total_trades': memory.total_trades
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_bot():
    """Start trading bot in background thread"""
    print("Starting trading_loop thread")
    
    def run_bot():
        bot_status['started'] = True
        bot_status['start_time'] = datetime.now()
        bot_status['thread_alive'] = True
        try:
            main()
        except Exception as e:
            print(f"Bot error: {e}")
            bot_status['thread_alive'] = False
    
    thread = threading.Thread(target=run_bot, name="TradingBot")
    thread.daemon = False  # ✅ Changed from True
    thread.start()
    print("Trading bot thread started (non-daemon)")

# Start bot when module loads
start_bot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=varport, threaded=True)