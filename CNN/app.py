from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# ─── LOAD MODEL & TOKENIZER ─────────────────────────────
model = tf.keras.models.load_model("cnn_lstm_final.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 600

# ─── CLEAN TEXT ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── PREDICT FUNCTION ───────────────────────────────────
def predict_text(text):
    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_LEN, padding="post"
    )

    prob = model.predict(padded)[0][0]
    label = "REAL" if prob > 0.5 else "FAKE"

    return prob, label

# ─── API ROUTE ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        prob, label = predict_text(text)

        return jsonify({
            "prediction": int(prob > 0.5),
            "label": label,
            "confidence": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── RUN SERVER ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)