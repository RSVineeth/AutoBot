from flask import Flask, Response
import os
import threading
from trading_loop import main  # import the function


varport = int(os.getenv("PORT", 5000))
app = Flask(__name__)

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return Response("Bot is running!", status=200)

def start_bot():
    thread = threading.Thread(target=main)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    start_bot()
    app.run(host="0.0.0.0", port=varport, threaded=True)
