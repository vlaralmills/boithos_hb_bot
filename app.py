
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import json
from pathlib import Path



load_dotenv()
app = Flask(__name__)
CORS(app)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HISTORY_FILE = "history.json"

# Φόρτωση ιστορικού ή δημιουργία αν δεν υπάρχει
if Path(HISTORY_FILE).exists():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
else:
    history = {}

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user = request.remote_addr
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Δεν έλαβα ερώτηση."}), 400

    messages = history.get(user, [])
    messages.append({"role": "user", "content": question})

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": answer})
    history[user] = messages[-10:]  # Κρατάμε μόνο τα τελευταία 10

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return jsonify({"answer": answer})

@app.route("/clear", methods=["POST"])
def clear():
    user = request.remote_addr
    history[user] = []
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "cleared"})

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>", methods=["GET"])
def serve_file(path):
    return send_from_directory(".", path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
