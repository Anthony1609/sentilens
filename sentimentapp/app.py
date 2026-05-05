"""
app.py — Sentiment Analysis Web App
Serves the frontend UI and REST API endpoints.
"""
import os, sys, json, logging, traceback
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from inference import SentimentPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
predictor = SentimentPredictor(model_dir=MODEL_DIR)

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html")

# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or not body.get("text", "").strip():
        return jsonify({"error": "Missing or empty 'text' field."}), 400
    try:
        result = predictor.predict(body["text"])
        return jsonify(result), 200
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    body = request.get_json(silent=True)
    if not body or not isinstance(body.get("texts"), list):
        return jsonify({"error": "Missing 'texts' list."}), 400
    if len(body["texts"]) > 100:
        return jsonify({"error": "Max 100 items per batch."}), 422
    try:
        return jsonify(predictor.predict_batch(body["texts"])), 200
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify(predictor.health()), 200

@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
