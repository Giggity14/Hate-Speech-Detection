# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re
import os

app = Flask(__name__)
CORS(app)   # Allow Chrome Extension to access the API

# Correct model path (model is inside the SAME hsd folder)
MODEL_PATH = "xlmr_hate_model"

print("Loading model... please wait.")
try:
    # Load tokenizer + model from local folder
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f" ERROR loading model: {e}")
    print(f"Make sure the folder '{MODEL_PATH}' exists in the hsd/ directory.")
    tokenizer, model = None, None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)  # keep Nepali characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_hate_logic(text):
    cleaned = clean_text(text)

    if not cleaned:
        return "NOT HATE", 0.0

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    probs = probs.squeeze().tolist()
    hate_prob = probs[1]  # index 1 = hate
    not_hate_prob = probs[0]

    label = "HATE SPEECH" if hate_prob > not_hate_prob else "NOT HATE"
    return label, hate_prob


@app.route('/predict', methods=['POST'])
def predict():
    if tokenizer is None or model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.json
    text = data.get('text', '')

    label, confidence = predict_hate_logic(text)

    return jsonify({
        "is_hate": label == "HATE SPEECH",
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
