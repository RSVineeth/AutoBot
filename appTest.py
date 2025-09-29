from flask import Flask, Response
import os
import threading
from trading_loop import main, memory  # import the function
import time
from datetime import datetime  


varport = int(os.getenv("PORT", 5000))
app = Flask(__name__)

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return Response("Bot is running!", status=200)

@app.route('/uptime', methods=['GET'])
def uptime():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return Response(f"Bot uptime check: {now}", status=200)


def start_bot():
    print("🚀 Starting trading_loop thread")
    thread = threading.Thread(target=main)
    # thread.daemon = True
    thread.daemon = False
    thread.start()

start_bot()

if __name__ == "__main__":
    # start_bot()
    app.run(host="0.0.0.0", port=varport, threaded=True)
