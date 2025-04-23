from flask import Flask, Response
import os

varport = int(os.getenv("PORT", 5000))

app = Flask(__name__)

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return Response("Bot is running!", status=200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=varport)
