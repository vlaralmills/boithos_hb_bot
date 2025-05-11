
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=openai_api_key)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with open("qa_data.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)
    questions = [q["question"] for q in qa_pairs]
    answers = [q["answer"] for q in qa_pairs]
    question_embeddings = model.encode(questions, convert_to_tensor=True)

import datetime

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json["question"]
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    hits = util.semantic_search(user_embedding, question_embeddings, top_k=1)[0]
    best_match = qa_pairs[hits[0]["corpus_id"]]

    prompt = f"Ερώτηση: {user_question}\nΓνωστή απάντηση: {best_match['answer']}\nΔώσε απάντηση:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    final_answer = response.choices[0].message.content

    # --- αποθήκευση στο history.json
    history_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": user_question,
        "answer": final_answer
    }
    try:
        with open("history.json", "r", encoding="utf-8") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append(history_entry)
    with open("history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return jsonify({"answer": final_answer})


from flask import send_from_directory

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/forma.html")
def serve_forma():
    return send_from_directory(".", "forma.html")

@app.route("/eme.pdf")
def serve_pdf():
    return send_from_directory(".", "eme.pdf")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
